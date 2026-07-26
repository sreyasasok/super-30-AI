import asyncio
import base64
import json
import re
from typing import List, Literal, Optional

import cv2
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import settings
from core.logging import get_logger
from services.broker import broker, redis_client
from services.vector_db import baseline_collection
from services.video_fetch import VideoFetchError, cleanup_video_source, resolve_video_source
from tasks.cv_engine import (
    DIRECTIONAL_PROFILES,
    HIGH_CONFIDENCE_MIN_VISIBILITY,
    TRACKING_PROFILE_FRIENDLY_NAMES,
    compute_metric_from_landmarks,
    detect_landmarks,
    find_bowling_release_frame,
    find_motion_event_frames,
    get_directional_label,
    get_player_adjustment_label,
    get_relevant_visibility,
)

logger = get_logger(__name__)
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=settings.OPENAI_MAX_RETRIES)

# Bounds how many events' triage+insight calls run concurrently. Shared at module level
# (not recreated per video) so multiple videos processed by the same worker don't each
# spin up their own uncapped concurrency on top of each other.
_event_semaphore = asyncio.Semaphore(settings.VIDEO_EVENT_CONCURRENCY)


class VisionTriageError(Exception):
    """Raised when the multimodal triage call fails or returns no parsed result."""


class BattingTriageSchema(BaseModel):
    assigned_profile: Literal[
        "FOOTWORK_WIDTH", "HAND_BACKLIFT", "SHOULDER_TILT", "FRONT_KNEE_BEND", "HEAD_STABILITY", "ELBOW_ELEVATION"
    ] = Field(..., description="The target tracking profile to pass to the MediaPipe PoseLandmarker engine.")
    primary_flaw: str = Field(..., description="Concise description of the primary mechanical error observed.")
    secondary_issues: List[str] = Field(..., description="Auxiliary technical flaws visible in the action frame.")
    additional_flagged_profile: Optional[Literal[
        "FOOTWORK_WIDTH", "HAND_BACKLIFT", "SHOULDER_TILT", "FRONT_KNEE_BEND", "HEAD_STABILITY", "ELBOW_ELEVATION"
    ]] = Field(
        default=None,
        description=(
            "A SECOND, clearly and separately visible technical issue on a DIFFERENT profile than "
            "assigned_profile — only set this if you can confidently see a genuinely distinct issue "
            "across the sequence, not a restatement of the primary one. Leave null if there's really "
            "only one issue; do not force a second finding."
        ),
    )
    additional_flaw: Optional[str] = Field(
        default=None,
        description="Required if additional_flagged_profile is set: concise description of that second issue.",
    )


class BowlingTriageSchema(BaseModel):
    assigned_profile: Literal["BOWLING_ARM_HEIGHT", "FRONT_KNEE_BRACE", "HEAD_STABILITY", "RELEASE_ALIGNMENT"] = (
        Field(..., description="The target tracking profile to pass to the MediaPipe PoseLandmarker engine.")
    )
    primary_flaw: str = Field(..., description="Concise description of the primary mechanical error observed.")
    secondary_issues: List[str] = Field(..., description="Auxiliary technical flaws visible in the action frame.")
    additional_flagged_profile: Optional[
        Literal["BOWLING_ARM_HEIGHT", "FRONT_KNEE_BRACE", "HEAD_STABILITY", "RELEASE_ALIGNMENT"]
    ] = Field(
        default=None,
        description=(
            "A SECOND, clearly and separately visible technical issue on a DIFFERENT profile than "
            "assigned_profile — only set this if you can confidently see a genuinely distinct issue "
            "across the sequence, not a restatement of the primary one. Leave null if there's really "
            "only one issue; do not force a second finding."
        ),
    )
    additional_flaw: Optional[str] = Field(
        default=None,
        description="Required if additional_flagged_profile is set: concise description of that second issue.",
    )


_TRIAGE_SCHEMAS_BY_DISCIPLINE = {
    "BATTING": BattingTriageSchema,
    "BOWLING": BowlingTriageSchema,
}

_TRIAGE_SYSTEM_PROMPTS_BY_DISCIPLINE = {
    "BATTING": (
        "You are an elite cricket batting biomechanics classifier. Analyze the provided frame(s) "
        "and map the primary movement error to exactly ONE profile: 'FOOTWORK_WIDTH', 'HAND_BACKLIFT', "
        "'SHOULDER_TILT', 'FRONT_KNEE_BEND', 'HEAD_STABILITY', or 'ELBOW_ELEVATION'. Identify any "
        "secondary faults as free text. Separately: if, and only if, you can clearly see a second, "
        "genuinely distinct issue on a DIFFERENT profile from the primary one — not just another "
        "way of describing the same flaw — set additional_flagged_profile and additional_flaw for "
        "it. Leave both null if there's really only one issue."
    ),
    "BOWLING": (
        "You are an elite cricket bowling biomechanics classifier. Analyze the provided frame(s) "
        "and map the primary movement error to exactly ONE profile: 'BOWLING_ARM_HEIGHT', "
        "'FRONT_KNEE_BRACE', 'HEAD_STABILITY', or 'RELEASE_ALIGNMENT'. Identify any secondary faults "
        "as free text. Separately: if, and only if, you can clearly see a second, genuinely distinct "
        "issue on a DIFFERENT profile from the primary one — not just another way of describing the "
        "same flaw — set additional_flagged_profile and additional_flaw for it. Leave both null if "
        "there's really only one issue."
    ),
}


class DisciplineClassificationSchema(BaseModel):
    discipline: Literal["BATTING", "BOWLING"] = Field(
        description="Whether the frame shows a batter or a bowler as the primary subject."
    )


async def _classify_discipline(frame, session_id: str) -> str:
    """Runs once per video (not per event) against the first detected event's frame, only
    when the app didn't already tell us the discipline. Falls back to BATTING — the more
    common practice-session case — if the call fails, rather than blocking the whole video."""
    vision_frame = _downscale_for_vision(frame)
    ok, buffer = cv2.imencode(".jpg", vision_frame)
    if not ok:
        logger.error("Failed to JPEG-encode frame for discipline classification | session_id=%s", session_id)
        return "BATTING"

    base64_image = base64.b64encode(buffer).decode("utf-8")
    try:
        response = await openai_client.beta.chat.completions.parse(
            model=settings.OPENAI_MODEL,
            reasoning_effort=settings.OPENAI_REASONING_EFFORT,
            response_format=DisciplineClassificationSchema,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Is the main cricket player in this frame batting or bowling? "
                                "Classify as exactly one of BATTING or BOWLING."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            timeout=12.0,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("discipline classification returned no parsed result")
        logger.info("Classified session discipline | session_id=%s discipline=%s", session_id, parsed.discipline)
        return parsed.discipline
    except Exception as exc:
        logger.error("Discipline classification failed, defaulting to BATTING | session_id=%s error=%s", session_id, exc)
        return "BATTING"


def _downscale_for_vision(frame):
    """Returns a resized copy for the vision API call only. Never mutates the original
    frame object — pose detection needs it at full resolution separately, and this
    function's return value is never assigned back over the caller's `frame`."""
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= settings.VISION_IMAGE_MAX_DIMENSION:
        return frame
    scale = settings.VISION_IMAGE_MAX_DIMENSION / longest_side
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


# Offsets (in seconds) of frames to include alongside the canonical event frame when asking the
# vision model to classify the flaw. A real-footage comparison found a short sequence lets the
# model reliably surface a second, motion-path-based issue (e.g. the bowling arm crossing the
# body) that's only visible across time, not in a single still — confirmed against the actual
# deterministic pose math that both the single-frame-visible issue AND the sequence-visible issue
# were simultaneously, genuinely present on the same delivery. Pose math itself is unaffected —
# it still only ever measures the single canonical frame; this only changes what the classifier
# sees.
TRIAGE_CLIP_OFFSETS_SECONDS = (-0.2, -0.1, 0.0, 0.1, 0.2)


def _extract_clip_frames(video_path: str, center_frame_index: int, fps: float) -> list:
    """Reads the short sequence of frames around center_frame_index described by
    TRIAGE_CLIP_OFFSETS_SECONDS, for the vision-triage call only. Falls back to an empty list
    if none could be read (caller is responsible for falling back to the canonical single
    frame in that case) rather than raising — a clip-extraction hiccup shouldn't cancel the
    event the way a missing canonical frame would."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for offset_seconds in TRIAGE_CLIP_OFFSETS_SECONDS:
        target = max(0, center_frame_index + round(offset_seconds * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    return frames


async def _run_vision_triage(frames: list, session_id: str, discipline: str):
    """Classifies the tracking profile for a short sequence of BGR frames (see
    TRIAGE_CLIP_OFFSETS_SECONDS — may be a single frame if clip extraction found nothing else)
    via structured output, using the schema/prompt for the given discipline (BATTING or
    BOWLING) so the model can only pick a profile that's actually valid for that discipline."""
    schema = _TRIAGE_SCHEMAS_BY_DISCIPLINE[discipline]
    system_prompt = _TRIAGE_SYSTEM_PROMPTS_BY_DISCIPLINE[discipline]

    image_blocks = []
    for frame in frames:
        vision_frame = _downscale_for_vision(frame)
        ok, buffer = cv2.imencode(".jpg", vision_frame)
        if not ok:
            raise VisionTriageError("Failed to JPEG-encode a target frame.")
        base64_image = base64.b64encode(buffer).decode("utf-8")
        image_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    instruction = (
        "Extract technical error profile for MediaPipe routing."
        if len(image_blocks) == 1
        else (
            f"These {len(image_blocks)} frames are sequential, spanning shortly before, during, and "
            "after the action. Extract the primary technical error profile for MediaPipe routing, "
            "using the sequence for motion context — and check whether a second, distinct issue is "
            "visible across the sequence even if it isn't obvious in any single frame."
        )
    )

    try:
        response = await openai_client.beta.chat.completions.parse(
            model=settings.OPENAI_MODEL,
            reasoning_effort=settings.OPENAI_REASONING_EFFORT,
            response_format=schema,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": instruction}, *image_blocks],
                },
            ],
            timeout=15.0,
        )
    except Exception as exc:
        raise VisionTriageError(f"Vision triage API call failed: {exc}") from exc

    parsed = response.choices[0].message.parsed
    if parsed is None:
        refusal = response.choices[0].message.refusal
        raise VisionTriageError(f"Vision triage returned no parsed result (refusal={refusal!r}).")

    logger.info(
        "Vision triage classified profile | session_id=%s profile=%s additional_profile=%s",
        session_id, parsed.assigned_profile, parsed.additional_flagged_profile,
    )
    return parsed


# Strips a leftover intensifier immediately before the {direction} token (e.g. "too {direction}"
# -> "{direction}") before substitution. The token already expands to the full phrase (e.g. "too
# high"), so a qualifier the model left next to it despite being told not to would otherwise
# double up ("too too high") once substituted. Belt-and-suspenders alongside the prompt
# instruction, matching this file's existing pattern of never trusting prompt adherence alone.
_DIRECTION_QUALIFIER_PATTERN = re.compile(r"\b(too|very|quite|a bit|a little)\s+(?=\{direction\})", re.IGNORECASE)


class VideoInsightSchema(BaseModel):
    coach_summary: str = Field(
        description="One precise sentence for a coach: names the metric and the measured deviation."
    )
    player_friendly_summary: str = Field(
        description=(
            "One encouraging sentence a young player or their parent can understand — no jargon, "
            "profile codes, or percentages, just what to work on."
        )
    )
    corrected_primary_flaw: str = Field(
        description=(
            "The original primary_flaw description, rewritten so any directional claim (high/low/"
            "left/right) matches the measured direction instead of the vision model's own guess."
        )
    )


async def _generate_video_insight(
    assigned_profile: str,
    primary_flaw: str,
    secondary_issues: List[str],
    measured_metric: float,
    session_id: str,
) -> tuple:
    """Returns (coach_summary, player_friendly_summary, corrected_primary_flaw). Falls back to
    grounded, jargon-free templates — built from the real triage/pose output, never invented
    text — if the LLM call fails or returns nothing usable.

    corrected_primary_flaw exists because the vision-triage call that originally wrote
    `primary_flaw` never sees the actual computed measured_metric — it's a free-text guess from
    pixels alone, generated before pose math even runs. A real-footage comparison found it can
    flatly contradict the deterministic, pose-landmark-derived direction (e.g. triage says "arm
    too low" while the measured wrist/shoulder position says "too high" for the same frame, which
    a visual check confirmed was the correct read). Rather than trust either model's own
    directional language, this call rewrites primary_flaw using the same token-substitution
    trick as coach_summary, so the direction word is always the deterministic one, never a guess."""
    friendly_name = TRACKING_PROFILE_FRIENDLY_NAMES.get(assigned_profile, assigned_profile)
    fallback_coach = f"Flaw detected: {primary_flaw} ({abs(measured_metric):.2f}% deviation)."
    fallback_player = f"Let's work on your {friendly_name} — talk it through with your coach."
    fallback_primary_flaw = primary_flaw

    # For directional profiles, compute exact labels for both current and baseline values in Python
    # first. We instruct the model to use the literal tokens `{direction}` and `{player_adjustment}`
    # in its summary, replacing them programmatically in Python afterward. This prevents the model's
    # semantic biases from overriding or contradicting the actual mathematical signs.
    direction_label = ""
    adjustment_label = ""
    player_instruction = ""
    if assigned_profile in DIRECTIONAL_PROFILES:
        direction_label = get_directional_label(assigned_profile, measured_metric)
        adjustment_label = get_player_adjustment_label(assigned_profile, measured_metric)
        metric_line = (
            f"Metric: {assigned_profile} ({friendly_name}). "
            f"Measured deviation magnitude: {abs(measured_metric):.2f} units. "
            f"Direction is physically '{direction_label}' (above/high is negative, below/low is positive)."
        )
        player_instruction = (
            f"IMPORTANT: You MUST use the literal token '{{player_adjustment}}' in your player sentence where the "
            "action word (e.g. raising/lowering/moving left/moving right) belongs (e.g., 'focus on {player_adjustment} your "
            "bowling arm'). Do NOT write the physical correction word yourself."
        )
    else:
        metric_line = (
            f"Metric: {assigned_profile} ({friendly_name}) — NOT a directional metric. "
            f"Measured deviation: {measured_metric:.2f}%. Do NOT use the words 'high', 'low', 'left', or "
            "'right' to describe this — describe the deviation only by its size/percentage."
        )
        player_instruction = (
            "Speak directly to the player about focusing on standard form for this metric. "
            "This is NOT a directional metric: do not use the literal token '{direction}' or "
            "'{player_adjustment}' anywhere in either summary — write plain, complete sentences "
            "with no placeholders."
        )

    try:
        response = await openai_client.beta.chat.completions.parse(
            model=settings.OPENAI_MODEL,
            reasoning_effort=settings.OPENAI_REASONING_EFFORT,
            response_format=VideoInsightSchema,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "A cricket coaching dashboard needs two summaries of the same flaw finding.\n"
                        f"{metric_line} Primary flaw: '{primary_flaw}'. Secondary issues: "
                        f"{secondary_issues}.\n"
                        "1) coach_summary: one precise sentence for a coach, referencing the metric and "
                        "the deviation. The metric line above already tells you whether this is a directional "
                        "metric or not — follow that, don't guess. IMPORTANT: ONLY if the metric line says this "
                        "IS a directional metric, you MUST use the literal token '{direction}' in your sentence "
                        "where the direction (e.g. high/low/left/right) belongs (e.g., 'indicating the bowling "
                        "arm is {direction} at release'), and do NOT write the direction word yourself. If the "
                        "metric line says this is NOT directional, never use that token at all — describe the "
                        "deviation using only the percentage.\n"
                        "2) player_friendly_summary: one encouraging sentence a young player or their "
                        "parent can understand with no jargon, profile codes, or percentages — just what "
                        "changed and what to work on. "
                        f"{player_instruction}\n"
                        "3) corrected_primary_flaw: rewrite the original primary flaw description above to "
                        "fix any directional claim in it — the description was written before the actual "
                        "measurement was known, so any 'high'/'low'/'left'/'right' language in it may be wrong. "
                        "IMPORTANT: if the metric line says this IS directional, replace every directional "
                        "phrase in the rewritten description with the literal token '{direction}' — the token "
                        "already means the full phrase (e.g. 'too high'), so remove any qualifier word right "
                        "next to it too ('too', 'very', 'quite', 'a bit') and leave ONLY the bare token in that "
                        "spot (e.g. 'arm is {direction} at release', never 'arm is too {direction} at release'). "
                        "There may be more than one directional phrase — replace all of them the same way. If "
                        "the metric line says NOT directional, remove any high/low/left/right language "
                        "entirely and never use that token. Keep the rest of the original description's "
                        "content and tone intact — this is a correction, not a rewrite from scratch."
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
        corrected_primary_flaw = parsed.corrected_primary_flaw

        if assigned_profile in DIRECTIONAL_PROFILES:
            if "{direction}" in coach_summary:
                coach_summary = coach_summary.replace("{direction}", direction_label)
            else:
                coach_summary = f"{coach_summary.rstrip('.')} ({direction_label})."

            if "{player_adjustment}" in player_friendly_summary:
                player_friendly_summary = player_friendly_summary.replace("{player_adjustment}", adjustment_label)
            else:
                player_friendly_summary = f"{player_friendly_summary.rstrip('.')} (correction: {adjustment_label})."

            if "{direction}" in corrected_primary_flaw:
                corrected_primary_flaw = _DIRECTION_QUALIFIER_PATTERN.sub("", corrected_primary_flaw)
                corrected_primary_flaw = corrected_primary_flaw.replace("{direction}", direction_label)
            else:
                # Model didn't use the token even though asked -- can't guarantee the direction
                # words it did write are correct, so fall back to the original triage text rather
                # than ship a "corrected" flaw that was never actually grounded in the token swap.
                logger.warning(
                    "corrected_primary_flaw missing {direction} token, using original triage text | "
                    "session_id=%s profile=%s", session_id, assigned_profile,
                )
                corrected_primary_flaw = fallback_primary_flaw
        else:
            # A non-directional profile's summary should never contain a direction token — the
            # prompt tells the model not to. If it ignores that anyway, naively stripping the
            # token leaves a grammatically broken sentence. Rather than patch broken text, treat
            # it as a malformed response and use the deterministic, always-grammatical fallback.
            reserved_tokens = ("{direction}", "{player_adjustment}", "{current_direction}", "{baseline_direction}")
            if any(token in coach_summary or token in player_friendly_summary for token in reserved_tokens):
                logger.warning(
                    "Non-directional video insight leaked a direction token, using fallback | "
                    "session_id=%s profile=%s", session_id, assigned_profile,
                )
                return fallback_coach, fallback_player, fallback_primary_flaw
            if any(token in corrected_primary_flaw for token in reserved_tokens):
                corrected_primary_flaw = fallback_primary_flaw

        return coach_summary, player_friendly_summary, corrected_primary_flaw
    except Exception as exc:
        logger.error("OpenAI summary call failed | session_id=%s error=%s", session_id, exc)
        return fallback_coach, fallback_player, fallback_primary_flaw


def _check_regression_against_baseline(player_id: str, profile: str, measured_metric: float):
    """Returns (regression_detected, baseline_info) for a given player+profile+measurement,
    querying baseline_collection. Factored out so the primary finding and any additional
    finding both run identical regression logic instead of it drifting between two inlined
    copies."""
    regression_detected = False
    baseline_info = None
    try:
        baselines = baseline_collection.get(
            where={"$and": [{"player_id": player_id}, {"tracking_profile": profile}]}
        )
        if baselines and baselines["ids"] and len(baselines["ids"]) > 0:
            meta = baselines["metadatas"][0]
            baseline_val = float(meta["baseline_value"])
            issue_name = meta["issue_name"]
            resolved_at = meta.get("resolved_at", "")

            if baseline_val != 0:
                variance = abs(baseline_val - measured_metric) / abs(baseline_val)
                if variance > settings.REGRESSION_THRESHOLD:
                    regression_detected = True
                    deviation_percentage = round(variance * 100, 1)
                    baseline_info = {
                        "issue_name": issue_name,
                        "deviation_percentage": deviation_percentage,
                        "resolved_at": resolved_at,
                        "baseline_value": baseline_val,
                    }
    except Exception as e:
        logger.error("Failed to check player baseline regression | player_id=%s profile=%s error=%s", player_id, profile, e)
    return regression_detected, baseline_info


async def _build_finding(
    profile: str,
    flaw_text: str,
    secondary_issues: List[str],
    measured_metric: float,
    visibility: float,
    player_id: str,
    session_id: str,
) -> dict:
    """Runs the regression check + insight generation for one profile's measurement and
    returns a fully-formed finding dict (same shape whether it ends up as the top-level
    finding or an entry in additional_findings), so both call sites build identical
    structures instead of two copies that could drift apart.

    A regression alert ("a previously-fixed habit is returning") is a stronger claim than a
    routine flaw card, so it only fires when `visibility` clears HIGH_CONFIDENCE_MIN_VISIBILITY
    — stricter than the base MIN_LANDMARK_VISIBILITY that already gated whether this finding
    exists at all. Below that bar, the finding is still reported (the measurement passed the
    base gate), just without the regression escalation layered on top."""
    if visibility >= HIGH_CONFIDENCE_MIN_VISIBILITY:
        regression_detected, baseline_info = _check_regression_against_baseline(player_id, profile, measured_metric)
    else:
        regression_detected, baseline_info = False, None
        logger.info(
            "Skipping regression check — visibility below high-confidence bar | "
            "session_id=%s profile=%s visibility=%.2f", session_id, profile, visibility,
        )

    ai_insight, player_friendly_summary, flaw_text = await _generate_video_insight(
        profile, flaw_text, secondary_issues, measured_metric, session_id
    )

    if regression_detected and baseline_info:
        deviation_pct = baseline_info["deviation_percentage"]
        issue_name = baseline_info["issue_name"]
        resolved_suffix = f" (marked fixed on {baseline_info['resolved_at']})" if baseline_info["resolved_at"] else " (previously marked fixed)"
        ai_insight = f"AI Alert: Mechanical Regression Detected. Player is deviating from the Fixed Reference Video baseline by {deviation_pct:.1f}%. Habit returning: {issue_name}. {ai_insight}"
        player_friendly_summary = f"Your old habit of '{issue_name}'{resolved_suffix} is starting to show up a bit. Let's work to get it back to baseline! {player_friendly_summary}"

    finding = {
        "tracking_profile": profile,
        "measured_metric": round(measured_metric, 2),
        "primary_flaw": flaw_text,
        "ai_insight_summary": ai_insight,
        "player_friendly_summary": player_friendly_summary,
    }
    if regression_detected and baseline_info:
        finding["regression_detected"] = True
        finding["regression_info"] = baseline_info
    return finding


async def _process_event(
    event_index: int,
    frame_index: int,
    frame,
    clip_frames: list,
    fps: float,
    session_id: str,
    player_id: str,
    total_events: int,
    discipline: str,
) -> None:
    """Runs triage -> pose math -> insight -> publish for one detected motion event.
    Bounded by the module-level semaphore so many events don't all fire their OpenAI
    calls at once. A failure here (e.g. vision triage erroring out after retries are
    exhausted) logs and returns rather than raising — one bad event shouldn't cancel
    the other concurrently-running events' results via asyncio.gather.

    `clip_frames` (a short sequence around `frame_index`, see TRIAGE_CLIP_OFFSETS_SECONDS)
    is used only for the vision-triage classification call, so the model can pick up on a
    second, motion-path-based issue a single still can't show. `frame` (the one canonical,
    already-selected instant) is still what all pose math is measured from, for both the
    primary and any additional finding — multi-frame input never changes what's actually
    measured, only what the classifier is shown."""
    async with _event_semaphore:
        target_event_seconds = frame_index / fps

        try:
            triage = await _run_vision_triage(clip_frames or [frame], session_id, discipline)
        except VisionTriageError as exc:
            logger.error(
                "Vision triage failed | session_id=%s event=%s/%s error=%s",
                session_id,
                event_index + 1,
                total_events,
                exc,
            )
            return

        assigned_profile = triage.assigned_profile
        primary_flaw = triage.primary_flaw
        secondary_issues = triage.secondary_issues

        landmarks = detect_landmarks(frame)
        if landmarks is None:
            # Pose detection genuinely failed on this frame — publishing a flaw card with
            # a fabricated-looking "0.0" would misrepresent a failed measurement as a real
            # zero-deviation one. Skip this event rather than report an unquantified guess
            # as if it were a scored finding.
            logger.warning(
                "Skipping event — pose measurement failed | session_id=%s event=%s/%s profile=%s",
                session_id,
                event_index + 1,
                total_events,
                assigned_profile,
            )
            return

        measured_metric = compute_metric_from_landmarks(landmarks, assigned_profile)
        if measured_metric is None:
            logger.warning(
                "Skipping event — pose measurement failed | session_id=%s event=%s/%s profile=%s",
                session_id,
                event_index + 1,
                total_events,
                assigned_profile,
            )
            return

        primary_visibility = get_relevant_visibility(landmarks, assigned_profile)
        primary_finding = await _build_finding(
            assigned_profile, primary_flaw, secondary_issues, measured_metric, primary_visibility, player_id, session_id
        )

        # A second, genuinely distinct issue the triage call flagged from the motion sequence.
        # Reuses the SAME detected landmarks (no extra MediaPipe call) so both findings are
        # measured from the identical instant. Scoped to at most one additional finding for
        # now — real-footage testing showed this catches the concrete case that motivated it
        # (a coexisting motion-path issue like release alignment alongside an instantaneous
        # one like arm height) without open-ended cost/complexity.
        #
        # Gated on HIGH_CONFIDENCE_MIN_VISIBILITY, not just the base MIN_LANDMARK_VISIBILITY
        # that compute_metric_from_landmarks already applied — a second simultaneous finding
        # is a bigger claim on a coach's trust than the routine primary one, so it only gets
        # reported at decidedly-more-likely-than-not confidence, not just "not obviously broken".
        additional_findings = []
        additional_profile = triage.additional_flagged_profile
        additional_flaw = triage.additional_flaw
        if additional_profile and additional_profile != assigned_profile and additional_flaw:
            additional_metric = compute_metric_from_landmarks(landmarks, additional_profile)
            if additional_metric is None:
                logger.warning(
                    "Additional finding skipped — pose measurement failed | session_id=%s event=%s/%s profile=%s",
                    session_id, event_index + 1, total_events, additional_profile,
                )
            else:
                additional_visibility = get_relevant_visibility(landmarks, additional_profile)
                if additional_visibility < HIGH_CONFIDENCE_MIN_VISIBILITY:
                    logger.info(
                        "Additional finding suppressed — below high-confidence bar | "
                        "session_id=%s event=%s/%s profile=%s visibility=%.2f",
                        session_id, event_index + 1, total_events, additional_profile, additional_visibility,
                    )
                else:
                    additional_finding = await _build_finding(
                        additional_profile, additional_flaw, [], additional_metric,
                        additional_visibility, player_id, session_id,
                    )
                    additional_findings.append(additional_finding)

        shot_telemetry = {
            "session_id": session_id,
            "player_id": player_id,
            "event_index": event_index,
            "total_events": total_events,
            "discipline": discipline,
            "absolute_seconds": round(target_event_seconds, 2),
            "tracking_profile": primary_finding["tracking_profile"],
            "measured_metric": primary_finding["measured_metric"],
            "primary_flaw": primary_finding["primary_flaw"],
            "secondary_issues": secondary_issues,
            "ai_insight_summary": primary_finding["ai_insight_summary"],
            "player_friendly_summary": primary_finding["player_friendly_summary"],
            "additional_findings": additional_findings,
            # Convenience view for a consumer that wants to render N equal-weight cards
            # without special-casing "the primary one, then check this other array" — there's
            # no real severity hierarchy between primary_finding and additional_findings
            # (the model just names one profile first), that split only exists for backward
            # compatibility with the pre-existing top-level fields above. Every entry here has
            # the identical shape (see _build_finding): tracking_profile, measured_metric,
            # primary_flaw, ai_insight_summary, player_friendly_summary, and optionally
            # regression_detected/regression_info.
            "all_findings": [primary_finding] + additional_findings,
        }

        if primary_finding.get("regression_detected"):
            shot_telemetry["regression_detected"] = True
            shot_telemetry["regression_info"] = primary_finding["regression_info"]

        await redis_client.publish("shot_updates", json.dumps(shot_telemetry))
        logger.info(
            "Published shot telemetry | session_id=%s event=%s/%s additional_findings=%s",
            session_id,
            event_index + 1,
            total_events,
            len(additional_findings),
        )


@broker.task
async def pipeline_agentic_video_analysis(
    video_path: str,
    session_id: str,
    player_id: str,
    discipline: str | None = None,
) -> None:
    """
    1. Scan the whole clip via OpenCV frame differencing to find every distinct motion
       event (e.g. each delivery in a full net session), not just one global peak.
    2. If the app didn't tell us the discipline, classify it once (BATTING or BOWLING)
       from the first event's frame — a session is one practice type end-to-end, so this
       runs once per video, not once per event.
    3. For each event, concurrently (bounded by VIDEO_EVENT_CONCURRENCY): multimodal LLM
       triage sees a short clip around the event (not just one frame — lets it notice a
       second, motion-path-based issue a single still can't show) to pick the
       discipline-appropriate profile(s) to track, MediaPipe pose math measures each one
       from the single canonical frame, and a dual coach/player summary is generated per
       finding.
    4. Publish one Redis message per event for the Node.js core backend to consume — the
       primary finding in the usual top-level fields, plus an `additional_findings` list
       (usually empty) for a second, genuinely distinct issue when one was confidently
       detected.

    Because events run concurrently, messages can arrive out of event_index order —
    consumers must key off event_index, not arrival order.

    `video_path` may be a local path or a public http(s) URL (e.g. an S3/GCS/CDN link). The
    download (if any) happens here, inside the worker, rather than in the API route handler —
    that keeps the route's 202 response instant regardless of video size, since this task
    already runs asynchronously off the request/response cycle.
    """
    try:
        local_path, is_temp = await resolve_video_source(video_path)
    except VideoFetchError as exc:
        logger.error("Failed to download remote video | session_id=%s error=%s", session_id, exc)
        return

    try:
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            logger.error("Could not open video | session_id=%s path=%s", session_id, video_path)
            cap.release()
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        events = find_motion_event_frames(local_path, fps)
        if not events:
            logger.error("No motion events detected | session_id=%s path=%s", session_id, video_path)
            return

        total_events = len(events)
        logger.info("Detected %s motion event(s) | session_id=%s", total_events, session_id)

        if discipline is None:
            _, first_frame = events[0]
            discipline = await _classify_discipline(first_frame, session_id)
        else:
            logger.info("Using app-supplied discipline | session_id=%s discipline=%s", session_id, discipline)

        if discipline == "BOWLING":
            # Raw frame-differencing can't tell run-up/follow-through motion from the release
            # itself — confirmed via a real-footage audit that some coarse events landed
            # mid-run-up instead of at release. Refine each one to the frame with peak wrist
            # elevation within a window, which is a real release-point signature.
            refined_events = []
            for frame_index, frame in events:
                refined = find_bowling_release_frame(local_path, frame_index, fps)
                refined_events.append(refined if refined is not None else (frame_index, frame))
            events = refined_events

        # Extracted up front (like the bowling refinement above), not inside _process_event,
        # so that function stays purely about frames-already-in-hand + LLM/math calls.
        clips = [_extract_clip_frames(local_path, frame_index, fps) for frame_index, _ in events]

        await asyncio.gather(*(
            _process_event(event_index, frame_index, frame, clip_frames, fps, session_id, player_id, total_events, discipline)
            for event_index, ((frame_index, frame), clip_frames) in enumerate(zip(events, clips))
        ))
    finally:
        cleanup_video_source(local_path, is_temp)
