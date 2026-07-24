import os
from typing import List, Literal, Optional

import cv2
from fastapi import APIRouter, status
from openai import OpenAI
from pydantic import BaseModel, Field

from api.errors import ApiError
from config import settings
from core.logging import get_logger
from services.vector_db import drill_collection, baseline_collection
from tasks.cv_engine import (
    DIRECTIONAL_PROFILES,
    TRACKING_PROFILE_FRIENDLY_NAMES,
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
    video_path: str = Field(min_length=1)
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


class RegressionCheckRequest(BaseModel):
    current_video_path: str = Field(min_length=1)
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


def _read_peak_frame_or_error(video_path: str):
    """Shared by every endpoint that needs the action frame from a video, so they all measure
    the same frame for the same clip instead of drifting between "middle frame" and "peak
    motion frame" strategies."""
    if not os.path.isfile(video_path):
        raise ApiError(status.HTTP_404_NOT_FOUND, "NOT_FOUND", f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ApiError(status.HTTP_400_BAD_REQUEST, "INVALID_VIDEO", "Unable to read target video file format.")

    try:
        peak = find_peak_motion_frame(cap)
    finally:
        cap.release()

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
            f"Metric: {tracking_profile} ({friendly_name}). Baseline value: {baseline_value:.2f}. "
            f"Current value: {current_value:.2f}. Deviation: {deviation_percentage:.1f}%."
        )
        player_instruction = "Speak directly to the player about focusing on standard form for this metric."

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
            model="gpt-4o-mini",
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
                        "the percentage, and explicitly warning that the resolved habit is returning. IMPORTANT: "
                        "If this is a directional metric, you MUST use the literal tokens '{baseline_direction}' and "
                        "'{current_direction}' in your sentence where the direction words belong (e.g., 'indicating the "
                        "bowling arm was {baseline_direction} but has now become {current_direction}'). Do NOT write the "
                        "direction words yourself.\n"
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
            # Strip any accidental template tokens from non-directional metrics
            for token in ("{direction}", "{player_adjustment}", "{current_direction}", "{baseline_direction}"):
                coach_summary = coach_summary.replace(token, "").replace("  ", " ")
                player_friendly_summary = player_friendly_summary.replace(token, "").replace("  ", " ")

        return coach_summary, player_friendly_summary
    except Exception as exc:
        logger.error("OpenAI regression insight call failed | error=%s", exc)
        return fallback_coach, fallback_player


@router.post("/video-analyses", status_code=status.HTTP_202_ACCEPTED)
async def analyze_video(payload: VideoAnalysisRequest):
    if not os.path.isfile(payload.video_path):
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
        batch_results[item.note_id] = _format_drill_matches(ids, distances, metadatas, item.limit)

    logger.info("Batch drill recommendations matched | items=%s", len(payload.items))

    return {
        "status": 200,
        "message": "Success",
        "data": {"results": batch_results},
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

    frame = _read_peak_frame_or_error(payload.current_video_path)
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

