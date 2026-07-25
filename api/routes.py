import json
import os
from typing import List, Literal, Optional

import cv2
from fastapi import APIRouter, status
from openai import OpenAI
from pydantic import BaseModel, Field

from api.errors import ApiError
from config import settings
from core.logging import get_logger
from services.vector_db import drill_collection, baseline_collection, coach_preference_collection
from services.video_fetch import VideoFetchError, cleanup_video_source, is_remote_url, resolve_video_source
from tasks.cv_engine import (
    DIRECTIONAL_PROFILES,
    PROFILE_RELEVANT_LANDMARKS,
    TRACKING_PROFILE_FRIENDLY_NAMES,
    apply_landmark_corrections,
    detect_landmarks,
    extract_named_landmarks,
    compute_metric_from_landmarks,
    find_peak_motion_frame,
    get_directional_label,
    get_player_adjustment_label,
    process_biomechanical_math,
)
from tasks.video_tasks import pipeline_agentic_video_analysis

router = APIRouter(prefix="/api/v1/ai")
logger = get_logger(__name__)
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=settings.OPENAI_MAX_RETRIES)


class RegressionInsightSchema(BaseModel):
    coach_summary: str = Field(
        description="One precise sentence for a coach: names the metric and the percentage deviation."
    )
    player_friendly_summary: str = Field(
        description=(
            "One encouraging sentence a young player or their parent can understand — no jargon, "
            "profile codes, or percentages, just what changed and what to work on."
        )
    )


TRACKING_PROFILES = Literal[
    # Batting
    "FOOTWORK_WIDTH",
    "HAND_BACKLIFT",
    "SHOULDER_TILT",
    "FRONT_KNEE_BEND",
    "HEAD_STABILITY",
    "ELBOW_ELEVATION",
    # Bowling
    "BOWLING_ARM_HEIGHT",
    "FRONT_KNEE_BRACE",
    "RELEASE_ALIGNMENT",
]


class VideoAnalysisRequest(BaseModel):
    video_path: str = Field(min_length=1, description="Local filesystem path or a public http(s) URL (e.g. an S3/GCS/CDN link).")
    session_id: str = Field(min_length=1)
    player_id: str = Field(min_length=1)
    discipline: Optional[Literal["BATTING", "BOWLING"]] = Field(
        default=None,
        description="If the app already knows the practice type, pass it here to skip an "
        "auto-classification call. Left null, the service classifies it from the footage.",
    )


class DrillRecommendationRequest(BaseModel):
    note_id: str
    fault_tag: str
    combined_search_text: str = Field(min_length=1)
    limit: int = Field(default=3, ge=1, le=10, description="Number of top matching drills to return")
    coach_id: Optional[str] = Field(
        default=None,
        description="If provided, boosts a drill this coach has previously picked for a similar "
        "flaw note (logged via /drill-recommendation-feedback) to the top of the results.",
    )


LANDMARK_NAMES = Literal[
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
]


class PoseLandmarksRequest(BaseModel):
    video_path: str = Field(min_length=1, description="Local filesystem path or a public http(s) URL (e.g. an S3/GCS/CDN link).")
    tracking_profile: Optional[TRACKING_PROFILES] = Field(
        default=None,
        description="If given, only the joints that profile's metric reads are returned. "
        "Omit to get all 10 tracked joints.",
    )


class LandmarkCorrection(BaseModel):
    landmark_name: LANDMARK_NAMES
    x: float = Field(description="Normalized x in [0, 1], same image-coordinate convention MediaPipe returns.")
    y: float = Field(description="Normalized y in [0, 1], origin top-left, increasing downward.")


class PoseCorrectionRequest(BaseModel):
    video_path: str = Field(min_length=1, description="Local filesystem path or a public http(s) URL (e.g. an S3/GCS/CDN link).")
    tracking_profile: TRACKING_PROFILES
    corrections: List[LandmarkCorrection] = Field(min_length=1, description="One or more joints the coach dragged into a corrected position.")
    purpose: Literal["BASELINE", "ROOT_CAUSE"] = Field(
        description="BASELINE: the corrected_value is meant to be registered as a Fixed Reference "
        "Baseline. ROOT_CAUSE: generates a narrative linking the correction to note_context."
    )
    note_context: Optional[str] = Field(
        default=None,
        description="The coach's symptom note (e.g. 'playing away from body'). Only used, and only "
        "produces a root_cause_summary, when purpose=ROOT_CAUSE.",
    )


class DrillRecommendationFeedbackRequest(BaseModel):
    note_id: str = Field(min_length=1)
    coach_id: str = Field(min_length=1)
    combined_search_text: str = Field(min_length=1, description="Same text that was queried for suggestions.")
    suggested_drill_ids: List[str] = Field(default_factory=list, description="What the AI proposed at the time.")
    selected_drill_id: str = Field(min_length=1, description="What the coach actually attached.")


class RegressionCheckRequest(BaseModel):
    current_video_path: str = Field(min_length=1, description="Local filesystem path or a public http(s) URL (e.g. an S3/GCS/CDN link).")
    tracking_profile: TRACKING_PROFILES
    # Not gt=0: DIRECTIONAL_PROFILES (e.g. BOWLING_ARM_HEIGHT) return signed measurements,
    # so a legitimate baseline can be negative (arm above shoulder) or zero (level with
    # it). Only exactly zero is actually invalid, since it'd divide-by-zero below —
    # checked explicitly in the endpoint rather than constrained here.
    baseline_value: float
    original_note_text: Optional[str] = None
    issue_name: str = Field(min_length=1, description="The name of the original physical/technical flaw (e.g., 'Front foot crossing over')")
    resolved_at: Optional[str] = Field(None, description="Optional date or timestamp when the coach marked the issue as fixed (e.g., '2026-06-12')")


class DrillItemSchema(BaseModel):
    drill_id: str = Field(min_length=1, description="Unique identifier for the drill")
    name: str = Field(min_length=1, description="Human-readable title of the drill")
    description: str = Field(
        min_length=10,
        description="Tactical description of what the drill fixes. Used to build vector embeddings.",
    )
    category: Optional[str] = Field(None, description="Optional tag like 'FOOTWORK', 'BAT_PATH', or 'BALANCE'")


class DrillBatchIngestRequest(BaseModel):
    drills: List[DrillItemSchema] = Field(min_length=1, description="List of drills to vectorize and index")


class BatchDrillRecommendationRequest(BaseModel):
    items: List[DrillRecommendationRequest] = Field(
        min_length=1, max_length=20, description="Flaw cards to generate drill matches for in one batch call."
    )


def _format_drill_matches(ids: list, distances: list, metadatas: list, limit: int) -> list:
    """Scores and filters one query's raw ChromaDB results. Shared by the single and batch
    recommendation endpoints so their similarity math and cutoff can never drift apart."""
    matches = []
    for drill_id, distance, metadata in zip(ids, distances, metadatas):
        if len(matches) >= limit:
            break
        # ChromaDB cosine distance = 1 - cosine_similarity; clamp the floor since
        # dissimilar text can push raw cosine similarity slightly negative.
        similarity = max(0.0, round(1 - distance, 2))
        if similarity >= settings.DRILL_SIMILARITY_CUTOFF:
            matches.append(
                {
                    "drill_id": drill_id,
                    "name": metadata.get("name", "Unknown Drill"),
                    "category": metadata.get("category"),
                    "similarity_score": similarity,
                }
            )
    return matches



# How many of a coach's past feedback entries to pull back when looking for their preference
# on a similar issue. Deliberately > 1: this is what makes the boost reflect what a coach
# actually prefers for a kind of issue (their most-frequently-chosen drill among similar past
# notes), rather than a coin-flip on which single past note happens to be worded closest to
# the new one.
COACH_PREFERENCE_LOOKBACK = 15


def _apply_personalization_boost(
    coach_id: Optional[str], combined_search_text: str, matches: list, limit: int
) -> list:
    """Boosts the drill a coach most often picks for issues similar to this one (logged via
    /drill-recommendation-feedback) to the front of `matches`. Shared by the single and batch
    recommendation endpoints, same rationale as _format_drill_matches.

    Pulls back the coach's COACH_PREFERENCE_LOOKBACK most similar past feedback entries, keeps
    only the ones that clear DRILL_SIMILARITY_CUTOFF (so a loosely-related past note can't
    count toward "what they prefer for this issue"), and picks whichever drill appears most
    often among those — not just whichever single past note is worded closest to the new one.
    Ties broken by highest similarity.

    Gated two ways so a coach's history can't misfire on an unrelated flaw:
    1. Text similarity against DRILL_SIMILARITY_CUTOFF on every counted instance (same
       constant standard matching uses).
    2. Category safety gate: the boosted drill only leads the list if its category matches
       the top standard match's category (or either is unset/GENERAL) — otherwise it's still
       appended (marked personalized) rather than discarded, but doesn't override a
       topically-stronger standard result.
    """
    if not coach_id or coach_preference_collection.count() == 0:
        return matches

    try:
        results = coach_preference_collection.query(
            query_texts=[combined_search_text],
            n_results=min(COACH_PREFERENCE_LOOKBACK, coach_preference_collection.count()),
            where={"coach_id": coach_id},
        )
    except Exception as exc:
        logger.error("Personalization lookup failed | coach_id=%s error=%s", coach_id, exc)
        return matches

    ids = results["ids"][0] if results["ids"] else []
    if not ids:
        return matches

    distances = results["distances"][0] if results["distances"] else [1.0] * len(ids)
    metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)

    # Only past instances that are themselves a strong match for the CURRENT issue count
    # toward "what this coach prefers for this kind of issue" — a coach's pick for an
    # unrelated flaw shouldn't pollute the tally just because it's in their history.
    pick_counts: dict[str, int] = {}
    best_similarity: dict[str, float] = {}
    for distance, metadata in zip(distances, metadatas):
        similarity = max(0.0, round(1 - distance, 2))
        if similarity < settings.DRILL_SIMILARITY_CUTOFF:
            continue
        drill_id = metadata.get("selected_drill_id")
        if not drill_id:
            continue
        pick_counts[drill_id] = pick_counts.get(drill_id, 0) + 1
        best_similarity[drill_id] = max(best_similarity.get(drill_id, 0.0), similarity)

    if not pick_counts:
        return matches

    # Most-frequently-picked drill wins; ties broken by whichever was textually closest.
    max_count = max(pick_counts.values())
    tied_drill_ids = [d for d, c in pick_counts.items() if c == max_count]
    preferred_drill_id = max(tied_drill_ids, key=lambda d: best_similarity[d])
    similarity = best_similarity[preferred_drill_id]
    times_preferred = pick_counts[preferred_drill_id]

    try:
        drill_lookup = drill_collection.get(ids=[preferred_drill_id])
    except Exception as exc:
        logger.error("Preferred drill lookup failed | drill_id=%s error=%s", preferred_drill_id, exc)
        return matches

    if not drill_lookup["ids"]:
        # Coach's previously-picked drill has since been deleted from the library.
        return matches

    drill_meta = drill_lookup["metadatas"][0]
    boosted = {
        "drill_id": preferred_drill_id,
        "name": drill_meta.get("name", "Unknown Drill"),
        "category": drill_meta.get("category"),
        "similarity_score": similarity,
        "personalized": True,
        "times_previously_selected": times_preferred,
    }

    top_category = matches[0]["category"] if matches else None
    categories_compatible = (
        not matches
        or not boosted["category"] or boosted["category"] == "GENERAL"
        or not top_category or top_category == "GENERAL"
        or boosted["category"] == top_category
    )

    deduped = [m for m in matches if m["drill_id"] != preferred_drill_id]
    result = [boosted] + deduped if categories_compatible else deduped + [boosted]
    return result[:limit]


async def _read_peak_frame_or_error(video_path: str):
    """Shared by every endpoint that needs the action frame from a video, so they all measure
    the same frame for the same clip instead of drifting between "middle frame" and "peak
    motion frame" strategies. `video_path` may be a local path or a public http(s) URL (e.g.
    an S3/GCS/CDN link) — resolved transparently, and any downloaded temp file is always
    cleaned up before returning, whether via the frame being found or an error being raised."""
    if not is_remote_url(video_path) and not os.path.isfile(video_path):
        raise ApiError(status.HTTP_404_NOT_FOUND, "NOT_FOUND", f"Video file not found: {video_path}")

    try:
        local_path, is_temp = await resolve_video_source(video_path)
    except VideoFetchError as exc:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, "VIDEO_DOWNLOAD_FAILED", str(exc))

    try:
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            cap.release()
            raise ApiError(status.HTTP_400_BAD_REQUEST, "INVALID_VIDEO", "Unable to read target video file format.")

        try:
            peak = find_peak_motion_frame(cap)
        finally:
            cap.release()
    finally:
        cleanup_video_source(local_path, is_temp)

    if peak is None:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, "UNPROCESSABLE_ENTITY", "Video has no readable frames.")

    _, frame = peak
    return frame


def _generate_regression_insight(
    tracking_profile: str,
    current_value: float,
    baseline_value: float,
    deviation_percentage: float,
    original_note_text: Optional[str],
    issue_name: str,
    resolved_at: Optional[str],
) -> tuple:
    """Returns (coach_summary, player_friendly_summary). Falls back to grounded, jargon-free
    templates — built from the real computed numbers, never invented ones — if the LLM call
    fails or returns nothing usable."""
    friendly_name = TRACKING_PROFILE_FRIENDLY_NAMES.get(tracking_profile, tracking_profile)

    resolved_suffix = f" (marked fixed on {resolved_at})" if resolved_at else " (previously marked fixed)"
    fallback_coach = (
        f"Regression alert: Previously resolved habit '{issue_name}'{resolved_suffix} has returned! "
        f"{tracking_profile} deviated by {deviation_percentage:.1f}% from historic baseline "
        f"({current_value:.2f} vs {baseline_value:.2f})."
    )
    fallback_player = (
        f"It looks like your old habit '{issue_name}' is starting to show up again. "
        f"Let's work with your coach to get your {friendly_name} back to your baseline form!"
    )

    # For directional profiles, compute exact labels for both current and baseline values in Python
    # first. We instruct the model to use the literal tokens `{current_direction}`,
    # `{baseline_direction}`, and `{player_adjustment}` in its summary, replacing them programmatically
    # in Python afterward. This prevents the model's semantic biases from overriding or contradicting
    # the actual mathematical signs.
    current_label = ""
    baseline_label = ""
    adjustment_label = ""
    player_instruction = ""
    if tracking_profile in DIRECTIONAL_PROFILES:
        current_label = get_directional_label(tracking_profile, current_value)
        baseline_label = get_directional_label(tracking_profile, baseline_value)
        adjustment_label = get_player_adjustment_label(tracking_profile, current_value)
        metric_line = (
            f"Metric: {tracking_profile} ({friendly_name}). "
            f"Baseline magnitude was {abs(baseline_value):.2f}. "
            f"Current magnitude is {abs(current_value):.2f}. "
            f"Baseline direction is physically '{baseline_label}', and current direction is physically '{current_label}'. "
            f"Deviation magnitude: {deviation_percentage:.1f}%."
        )
        player_instruction = (
            f"Speak directly to the player about the physical adjustment implied by shifting from baseline '{baseline_label}' "
            f"to current '{current_label}' (e.g. if shifting from 'too high' to 'too low', they must correct by 'raising' "
            "it; if shifting from 'too far left' to 'too far right', they must adjust 'left').\n"
            f"IMPORTANT: You MUST use the literal token '{{player_adjustment}}' in your player sentence where the "
            "action word (e.g. raising/lowering/moving left/moving right) belongs (e.g., 'focus on {player_adjustment} your "
            "bowling arm'). Do NOT write the physical correction word yourself."
        )
    else:
        metric_line = (
            f"Metric: {tracking_profile} ({friendly_name}) — NOT a directional metric. "
            f"Baseline value: {baseline_value:.2f}. Current value: {current_value:.2f}. "
            f"Deviation: {deviation_percentage:.1f}%. Do NOT use the words 'high', 'low', 'left', or "
            "'right' to describe this — describe the deviation only by its size/percentage."
        )
        player_instruction = (
            "Speak directly to the player about focusing on standard form for this metric. "
            "This is NOT a directional metric: do not use the literal tokens '{baseline_direction}', "
            "'{current_direction}', or '{player_adjustment}' anywhere in either summary — write plain, "
            "complete sentences with no placeholders."
        )

    # Context about the returning habit
    resolved_clause = f"which was marked as fixed on {resolved_at}" if resolved_at else "which was previously marked as fixed"
    habit_context = (
        f"CRITICAL CONTEXT: This is not just a general technique deviation. This is a MECHANICAL REGRESSION. "
        f"The player is slipping back into their old technical flaw: '{issue_name}', {resolved_clause}. "
        "Your summaries must explicitly warn the coach and the player/parent that this specific past, "
        "resolved habit is returning (e.g. 'Habit returning: [issue_name]')."
    )

    try:
        response = openai_client.beta.chat.completions.parse(
            model=settings.OPENAI_MODEL,
            reasoning_effort=settings.OPENAI_REASONING_EFFORT,
            response_format=RegressionInsightSchema,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "A cricket coaching dashboard needs two summaries of the same regression finding.\n"
                        f"{metric_line}\n"
                        f"{habit_context}\n"
                        f"Original coach note context: '{original_note_text or 'None'}'.\n"
                        "1) coach_summary: one precise sentence for a coach, referencing the metric name, "
                        "the percentage, and explicitly warning that the resolved habit is returning. The metric "
                        "line above already tells you whether this is a directional metric or not — follow that, "
                        "don't guess. IMPORTANT: ONLY if the metric line says this IS a directional metric, you "
                        "MUST use the literal tokens '{baseline_direction}' and '{current_direction}' in your "
                        "sentence where the direction words belong (e.g., 'indicating the bowling arm was "
                        "{baseline_direction} but has now become {current_direction}'), and do NOT write the "
                        "direction words yourself. If the metric line says this is NOT directional, never use "
                        "those tokens at all — describe the deviation using only the percentage.\n"
                        "2) player_friendly_summary: one encouraging sentence a young player or their "
                        "parent can understand with no jargon, profile codes, or percentages — just what "
                        "resolved habit is slipping back, and what to work on. "
                        f"{player_instruction}"
                    ),
                }
            ],
            timeout=10.0,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("structured insight response had no parsed result")

        coach_summary = parsed.coach_summary
        player_friendly_summary = parsed.player_friendly_summary
        if tracking_profile in DIRECTIONAL_PROFILES:
            if "{current_direction}" in coach_summary:
                coach_summary = coach_summary.replace("{current_direction}", current_label)
            else:
                coach_summary = f"{coach_summary.rstrip('.')} (current: {current_label})."

            if "{baseline_direction}" in coach_summary:
                coach_summary = coach_summary.replace("{baseline_direction}", baseline_label)
            else:
                coach_summary = f"{coach_summary.rstrip('.')} (baseline: {baseline_label})."

            if "{player_adjustment}" in player_friendly_summary:
                player_friendly_summary = player_friendly_summary.replace("{player_adjustment}", adjustment_label)
            else:
                player_friendly_summary = f"{player_friendly_summary.rstrip('.')} (correction: {adjustment_label})."
        else:
            # A non-directional profile's summary should never contain a direction token — the
            # prompt tells the model not to. If it ignores that anyway, naively stripping the
            # token leaves a grammatically broken sentence (e.g. "...was previously  but has
            # now become ,"). Rather than patch broken text, treat it as a malformed response
            # and use the deterministic, always-grammatical fallback instead.
            reserved_tokens = ("{direction}", "{player_adjustment}", "{current_direction}", "{baseline_direction}")
            if any(token in coach_summary or token in player_friendly_summary for token in reserved_tokens):
                logger.warning(
                    "Non-directional regression insight leaked a direction token, using fallback | profile=%s",
                    tracking_profile,
                )
                return fallback_coach, fallback_player

        return coach_summary, player_friendly_summary
    except Exception as exc:
        logger.error("OpenAI regression insight call failed | error=%s", exc)
        return fallback_coach, fallback_player


class RootCauseInsightSchema(BaseModel):
    root_cause_summary: str = Field(
        description="One precise sentence linking the coach's manual joint correction to the "
        "likely root cause of the symptom they described."
    )


def _generate_root_cause_summary(
    tracking_profile: str,
    original_value: float,
    corrected_value: float,
    deviation: float,
    note_context: str,
) -> str:
    """Returns a one-sentence root-cause narrative linking a coach's manual landmark
    correction (dragging a joint to where it should be) to the symptom they described.
    Falls back to a grounded, numbers-only template — never invented text — if the LLM
    call fails, matching the fallback rule every other insight helper in this file follows."""
    friendly_name = TRACKING_PROFILE_FRIENDLY_NAMES.get(tracking_profile, tracking_profile)
    fallback = (
        f"Correcting {friendly_name} shifts the measurement from {original_value:.2f} to "
        f"{corrected_value:.2f} ({deviation:+.2f}), consistent with the coach's note: '{note_context}'."
    )
    try:
        response = openai_client.beta.chat.completions.parse(
            model=settings.OPENAI_MODEL,
            reasoning_effort=settings.OPENAI_REASONING_EFFORT,
            response_format=RootCauseInsightSchema,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "A cricket coach manually corrected a player's joint position on a video "
                        f"frame to demonstrate proper form for the metric '{tracking_profile}' "
                        f"({friendly_name}). The AI's original measurement was {original_value:.2f}; "
                        f"the coach's corrected position measures {corrected_value:.2f} "
                        f"(a {deviation:+.2f} shift). The coach's note on the symptom they observed: "
                        f"'{note_context}'.\n"
                        "Write one precise sentence linking this symptom to the likely root cause "
                        "implied by the correction — e.g. if the note describes a downstream effect "
                        "(like playing away from the body), explain what the corrected joint position "
                        "suggests is the actual mechanical cause."
                    ),
                }
            ],
            timeout=10.0,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("root cause insight response had no parsed result")
        return parsed.root_cause_summary
    except Exception as exc:
        logger.error("OpenAI root cause insight call failed | error=%s", exc)
        return fallback


@router.post("/video-analyses", status_code=status.HTTP_202_ACCEPTED)
async def analyze_video(payload: VideoAnalysisRequest):
    # A URL is only validated for shape here — the actual download happens inside the
    # background task itself (pipeline_agentic_video_analysis), not in this handler, so a
    # large video's download time never delays the 202 response. A local path, by contrast,
    # can be checked for real right now, so we still fail fast on an obviously bad one.
    if not is_remote_url(payload.video_path) and not os.path.isfile(payload.video_path):
        raise ApiError(status.HTTP_404_NOT_FOUND, "NOT_FOUND", f"Video file not found: {payload.video_path}")

    await pipeline_agentic_video_analysis.kiq(
        payload.video_path, payload.session_id, payload.player_id, payload.discipline
    )
    logger.info("Queued video analysis | session_id=%s player_id=%s", payload.session_id, payload.player_id)

    return {
        "status": 202,
        "message": "Success",
        "data": {"queued": True, "session_id": payload.session_id},
    }


@router.post("/frames/pose-landmarks")
async def get_frame_pose_landmarks(payload: PoseLandmarksRequest):
    """Returns the AI-detected joint positions for a video's peak-motion frame, so the coach
    app can render them and let a coach drag one into a corrected position (see
    /pose-corrections)."""
    frame = await _read_peak_frame_or_error(payload.video_path)
    landmarks = extract_named_landmarks(frame)
    if landmarks is None:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "UNPROCESSABLE_ENTITY",
            "Could not detect pose landmarks in the video's action frame — try a clearer or closer clip.",
        )

    relevant = None
    if payload.tracking_profile is not None:
        relevant = PROFILE_RELEVANT_LANDMARKS[payload.tracking_profile]
        landmarks = {name: landmarks[name] for name in relevant}

    logger.info(
        "Extracted pose landmarks | video_path=%s profile=%s count=%s",
        payload.video_path, payload.tracking_profile, len(landmarks),
    )

    return {
        "status": 200,
        "message": "Success",
        "data": {"landmarks": landmarks, "relevant_to_profile": relevant},
    }


@router.post("/pose-corrections")
async def apply_pose_correction(payload: PoseCorrectionRequest):
    """Recomputes a tracking profile's metric using a coach's manually corrected joint
    position(s) instead of the AI's raw detection. purpose=BASELINE: the returned
    corrected_value is meant to be passed straight to POST /players/{player_id}/baselines.
    purpose=ROOT_CAUSE: pairs the correction with note_context to explain what the
    correction implies about the underlying technical cause of a described symptom."""
    frame = await _read_peak_frame_or_error(payload.video_path)
    landmarks = detect_landmarks(frame)
    if landmarks is None:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "UNPROCESSABLE_ENTITY",
            "Could not detect pose landmarks in the video's action frame — try a clearer or closer clip.",
        )

    original_value = compute_metric_from_landmarks(landmarks, payload.tracking_profile)
    if original_value is None:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "UNPROCESSABLE_ENTITY",
            f"Unable to compute a metric for profile {payload.tracking_profile}.",
        )

    corrections_map = {c.landmark_name: (c.x, c.y) for c in payload.corrections}
    corrected_landmarks = apply_landmark_corrections(landmarks, corrections_map)
    corrected_value = compute_metric_from_landmarks(corrected_landmarks, payload.tracking_profile)
    deviation = corrected_value - original_value

    root_cause_summary = None
    if payload.purpose == "ROOT_CAUSE" and payload.note_context:
        root_cause_summary = _generate_root_cause_summary(
            payload.tracking_profile, original_value, corrected_value, deviation, payload.note_context
        )

    logger.info(
        "Pose correction computed | video_path=%s profile=%s purpose=%s original=%.2f corrected=%.2f",
        payload.video_path, payload.tracking_profile, payload.purpose, original_value, corrected_value,
    )

    return {
        "status": 200,
        "message": "Success",
        "data": {
            "tracking_profile": payload.tracking_profile,
            "original_value": round(original_value, 2),
            "corrected_value": round(corrected_value, 2),
            "deviation_from_original": round(deviation, 2),
            "root_cause_summary": root_cause_summary,
        },
    }


@router.post("/drill-recommendations")
async def recommend_drills(payload: DrillRecommendationRequest):
    if drill_collection.count() == 0:
        logger.warning("Drill recommendation requested on empty collection | note_id=%s", payload.note_id)
        return {
            "status": 200,
            "message": "Success",
            "data": {"note_id": payload.note_id, "suggested_drills": []},
        }

    try:
        results = drill_collection.query(query_texts=[payload.combined_search_text], n_results=payload.limit)
    except Exception as exc:
        logger.error("Drill vector search failed | note_id=%s error=%s", payload.note_id, exc)
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "VECTOR_SEARCH_FAILED",
            "An error occurred while matching drills against the vector store.",
        )

    formatted_drills = []
    if results["ids"] and len(results["ids"][0]) > 0:
        formatted_drills = _format_drill_matches(
            results["ids"][0],
            results["distances"][0] if results["distances"] else [0.0] * len(results["ids"][0]),
            results["metadatas"][0],
            payload.limit,
        )

    formatted_drills = _apply_personalization_boost(
        payload.coach_id, payload.combined_search_text, formatted_drills, payload.limit
    )

    logger.info("Drill recommendations matched | note_id=%s count=%s", payload.note_id, len(formatted_drills))

    return {
        "status": 200,
        "message": "Success",
        "data": {"note_id": payload.note_id, "suggested_drills": formatted_drills},
    }


@router.post("/batch-drill-recommendations")
async def recommend_drills_batch(payload: BatchDrillRecommendationRequest):
    """Runs one ChromaDB multi-query call across several flaw cards instead of N round-trips."""
    if drill_collection.count() == 0:
        logger.warning("Batch drill recommendation requested on empty collection | items=%s", len(payload.items))
        return {
            "status": 200,
            "message": "Success",
            "data": {"results": {item.note_id: [] for item in payload.items}},
        }

    query_texts = [item.combined_search_text for item in payload.items]
    max_limit = max(item.limit for item in payload.items)

    try:
        results = drill_collection.query(query_texts=query_texts, n_results=max_limit)
    except Exception as exc:
        logger.error("Batch drill vector search failed | items=%s error=%s", len(payload.items), exc)
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "VECTOR_SEARCH_FAILED",
            "An error occurred while running batch drill recommendation matching.",
        )

    batch_results = {}
    for idx, item in enumerate(payload.items):
        ids = results["ids"][idx]
        distances = results["distances"][idx] if results["distances"] else [0.0] * len(ids)
        metadatas = results["metadatas"][idx]
        matches = _format_drill_matches(ids, distances, metadatas, item.limit)
        batch_results[item.note_id] = _apply_personalization_boost(
            item.coach_id, item.combined_search_text, matches, item.limit
        )

    logger.info("Batch drill recommendations matched | items=%s", len(payload.items))

    return {
        "status": 200,
        "message": "Success",
        "data": {"results": batch_results},
    }


@router.post("/drill-recommendation-feedback", status_code=status.HTTP_201_CREATED)
async def log_drill_recommendation_feedback(payload: DrillRecommendationFeedbackRequest):
    """Logs whether a coach accepted an AI drill suggestion or overrode it with a manual
    pick. This is the write side of the personalization loop that _apply_personalization_boost
    reads from — the same note_id+coach_id logged twice overwrites (upsert), so re-finalizing
    a draft note doesn't leave stale duplicate feedback behind."""
    accepted = payload.selected_drill_id in payload.suggested_drill_ids
    feedback_id = f"{payload.coach_id}_{payload.note_id}"
    metadata = {
        "coach_id": payload.coach_id,
        "note_id": payload.note_id,
        "suggested_drill_ids": json.dumps(payload.suggested_drill_ids),
        "selected_drill_id": payload.selected_drill_id,
        "accepted": accepted,
    }

    try:
        coach_preference_collection.upsert(
            ids=[feedback_id], documents=[payload.combined_search_text], metadatas=[metadata]
        )
    except Exception as exc:
        logger.error(
            "Drill recommendation feedback logging failed | coach_id=%s note_id=%s error=%s",
            payload.coach_id, payload.note_id, exc,
        )
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "FEEDBACK_LOGGING_FAILED",
            "An unexpected error occurred while logging drill recommendation feedback.",
        )

    logger.info(
        "Logged drill recommendation feedback | coach_id=%s note_id=%s accepted=%s",
        payload.coach_id, payload.note_id, accepted,
    )

    return {
        "status": 201,
        "message": "Success",
        "data": {
            "note_id": payload.note_id,
            "coach_id": payload.coach_id,
            "selected_drill_id": payload.selected_drill_id,
            "accepted": accepted,
        },
    }


@router.post("/drill-embeddings", status_code=status.HTTP_201_CREATED)
async def create_drill_embeddings(payload: DrillBatchIngestRequest):
    """Vectorizes and indexes drills into ChromaDB. Upsert semantics: existing drill_ids are updated in place."""
    ids = [d.drill_id for d in payload.drills]
    documents = [f"{d.name}: {d.description}" for d in payload.drills]
    metadatas = [{"name": d.name, "category": d.category or "GENERAL"} for d in payload.drills]

    try:
        drill_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    except Exception as exc:
        logger.error("Drill embedding upsert failed | drill_ids=%s error=%s", ids, exc)
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "VECTOR_INDEXING_FAILED",
            "An unexpected error occurred while vectorizing drill items.",
        )

    logger.info("Indexed drills into ChromaDB | count=%s", len(ids))

    return {
        "status": 201,
        "message": "Success",
        "data": {"indexed_count": len(ids), "drill_ids": ids},
    }


@router.get("/drills")
async def list_drills(limit: int = 100):
    """Lists all skills/drills stored in ChromaDB."""
    try:
        results = drill_collection.get(limit=limit)
    except Exception as exc:
        logger.error("Failed to retrieve drills from ChromaDB | error=%s", exc)
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "LIST_DRILLS_FAILED",
            "An unexpected error occurred while listing drills.",
        )

    formatted_drills = []
    if results and results["ids"]:
        for idx, drill_id in enumerate(results["ids"]):
            doc = results["documents"][idx] if results["documents"] else ""
            meta = results["metadatas"][idx] if results["metadatas"] else {}

            # The document is formatted as "Name: Description", let's split it back safely if possible
            desc = doc
            if ":" in doc:
                parts = doc.split(":", 1)
                desc = parts[1].strip()

            formatted_drills.append({
                "drill_id": drill_id,
                "name": meta.get("name", "Unknown Drill"),
                "description": desc,
                "category": meta.get("category", "GENERAL")
            })

    return {
        "status": 200,
        "message": "Success",
        "data": {"drills": formatted_drills}
    }


@router.delete("/drills/{drill_id}")
async def delete_drill(drill_id: str):
    """Deletes a specific skill/drill from the ChromaDB library."""
    try:
        drill_collection.delete(ids=[drill_id])
    except Exception as exc:
        logger.error("Failed to delete drill from ChromaDB | drill_id=%s error=%s", drill_id, exc)
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "DELETE_DRILL_FAILED",
            "An unexpected error occurred while deleting the drill.",
        )

    logger.info("Deleted drill from ChromaDB | drill_id=%s", drill_id)

    return {
        "status": 200,
        "message": "Success",
        "data": {"drill_id": drill_id, "deleted": True}
    }


@router.post("/regression-checks")
async def regression_check(payload: RegressionCheckRequest):
    if payload.baseline_value == 0:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "INVALID_REQUEST",
            "baseline_value cannot be zero (used as a division denominator).",
        )

    frame = await _read_peak_frame_or_error(payload.current_video_path)
    current_value = process_biomechanical_math(frame, payload.tracking_profile)
    if current_value is None:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "UNPROCESSABLE_ENTITY",
            "Could not detect pose landmarks in the video's action frame — try a clearer or closer clip.",
        )

    variance = abs(payload.baseline_value - current_value) / payload.baseline_value
    regression_detected = variance > settings.REGRESSION_THRESHOLD
    deviation_percentage = round(variance * 100, 1)

    insight_text = None
    player_friendly_summary = None
    if regression_detected:
        insight_text, player_friendly_summary = _generate_regression_insight(
            payload.tracking_profile,
            current_value,
            payload.baseline_value,
            deviation_percentage,
            payload.original_note_text,
            payload.issue_name,
            payload.resolved_at,
        )

    logger.info(
        "Regression check completed | profile=%s baseline=%s current=%.2f deviation=%.1f%% detected=%s",
        payload.tracking_profile,
        payload.baseline_value,
        current_value,
        deviation_percentage,
        regression_detected,
    )

    return {
        "status": 200,
        "message": "Success",
        "data": {
            "regression_detected": regression_detected,
            "tracking_profile": payload.tracking_profile,
            "baseline_value": payload.baseline_value,
            "current_value": round(current_value, 2),
            "deviation_percentage": deviation_percentage,
            "issue_name": payload.issue_name,
            "resolved_at": payload.resolved_at,
            "insight_text": insight_text,
            "player_friendly_summary": player_friendly_summary,
        },
    }


class BaselineRegisterRequest(BaseModel):
    baseline_id: str = Field(min_length=1, description="Unique identifier for this baseline record")
    tracking_profile: TRACKING_PROFILES
    baseline_value: float
    issue_name: str = Field(min_length=1, description="The name of the original physical/technical flaw (e.g. 'Dropped bowling arm')")
    original_note_text: Optional[str] = None
    resolved_at: Optional[str] = Field(None, description="Date when the issue was marked as fixed (e.g., '2026-06-12')")


@router.post("/players/{player_id}/baselines", status_code=status.HTTP_201_CREATED)
async def register_player_baseline(player_id: str, payload: BaselineRegisterRequest):
    """Registers a resolved technical issue as a 'Fixed Reference Baseline' for a player.
    This indexes the issue description into ChromaDB and saves the biomechanical joint value."""
    document = f"{payload.issue_name}: {payload.original_note_text or ''}"
    metadata = {
        "player_id": player_id,
        "tracking_profile": payload.tracking_profile,
        "baseline_value": payload.baseline_value,
        "issue_name": payload.issue_name,
        "original_note_text": payload.original_note_text or "",
        "resolved_at": payload.resolved_at or "",
    }

    try:
        baseline_collection.upsert(
            ids=[payload.baseline_id],
            documents=[document],
            metadatas=[metadata]
        )
    except Exception as exc:
        logger.error("Player baseline registration failed | player_id=%s baseline_id=%s error=%s", player_id, payload.baseline_id, exc)
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "BASELINE_REGISTRATION_FAILED",
            "An unexpected error occurred while registering the player reference baseline.",
        )

    logger.info("Registered player baseline | player_id=%s baseline_id=%s profile=%s value=%.2f", player_id, payload.baseline_id, payload.tracking_profile, payload.baseline_value)

    return {
        "status": 201,
        "message": "Success",
        "data": {
            "baseline_id": payload.baseline_id,
            "player_id": player_id,
            "tracking_profile": payload.tracking_profile,
            "baseline_value": payload.baseline_value,
            "issue_name": payload.issue_name,
        }
    }


@router.get("/players/{player_id}/baselines")
async def list_player_baselines(player_id: str):
    """Lists all registered 'Fixed Reference Baselines' for a specific player."""
    try:
        results = baseline_collection.get(where={"player_id": player_id})
    except Exception as exc:
        logger.error("Failed to query player baselines | player_id=%s error=%s", player_id, exc)
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "QUERY_BASELINES_FAILED",
            "An error occurred while listing player reference baselines.",
        )

    formatted = []
    if results and results["ids"]:
        for idx, id_val in enumerate(results["ids"]):
            meta = results["metadatas"][idx]
            formatted.append({
                "baseline_id": id_val,
                "tracking_profile": meta.get("tracking_profile"),
                "baseline_value": meta.get("baseline_value"),
                "issue_name": meta.get("issue_name"),
                "original_note_text": meta.get("original_note_text"),
                "resolved_at": meta.get("resolved_at"),
            })

    return {
        "status": 200,
        "message": "Success",
        "data": {"player_id": player_id, "baselines": formatted}
    }


@router.delete("/players/{player_id}/baselines/{baseline_id}")
async def delete_player_baseline(player_id: str, baseline_id: str):
    """Deletes a specific reference baseline for a player."""
    try:
        baseline_collection.delete(ids=[baseline_id])
    except Exception as exc:
        logger.error("Failed to delete baseline | player_id=%s baseline_id=%s error=%s", player_id, baseline_id, exc)
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "DELETE_BASELINE_FAILED",
            "An unexpected error occurred while deleting the player reference baseline.",
        )

    logger.info("Deleted player baseline | player_id=%s baseline_id=%s", player_id, baseline_id)

    return {
        "status": 200,
        "message": "Success",
        "data": {"baseline_id": baseline_id, "deleted": True}
    }

