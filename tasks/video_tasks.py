import asyncio
import base64
import json
from typing import List, Literal

import cv2
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import settings
from core.logging import get_logger
from services.broker import broker, redis_client
from services.vector_db import baseline_collection
from tasks.cv_engine import (
    DIRECTIONAL_PROFILES,
    TRACKING_PROFILE_FRIENDLY_NAMES,
    find_bowling_release_frame,
    find_motion_event_frames,
    get_directional_label,
    get_player_adjustment_label,
    process_biomechanical_math,
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


class BowlingTriageSchema(BaseModel):
    assigned_profile: Literal["BOWLING_ARM_HEIGHT", "FRONT_KNEE_BRACE", "HEAD_STABILITY", "RELEASE_ALIGNMENT"] = (
        Field(..., description="The target tracking profile to pass to the MediaPipe PoseLandmarker engine.")
    )
    primary_flaw: str = Field(..., description="Concise description of the primary mechanical error observed.")
    secondary_issues: List[str] = Field(..., description="Auxiliary technical flaws visible in the action frame.")


_TRIAGE_SCHEMAS_BY_DISCIPLINE = {
    "BATTING": BattingTriageSchema,
    "BOWLING": BowlingTriageSchema,
}

_TRIAGE_SYSTEM_PROMPTS_BY_DISCIPLINE = {
    "BATTING": (
        "You are an elite cricket batting biomechanics classifier. Analyze the provided frame "
        "and map the movement error to exactly ONE profile: 'FOOTWORK_WIDTH', 'HAND_BACKLIFT', "
        "'SHOULDER_TILT', 'FRONT_KNEE_BEND', 'HEAD_STABILITY', or 'ELBOW_ELEVATION'. Identify any "
        "secondary faults."
    ),
    "BOWLING": (
        "You are an elite cricket bowling biomechanics classifier. Analyze the provided frame "
        "and map the movement error to exactly ONE profile: 'BOWLING_ARM_HEIGHT', "
        "'FRONT_KNEE_BRACE', 'HEAD_STABILITY', or 'RELEASE_ALIGNMENT'. Identify any secondary faults."
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
            model="gpt-4o-mini",
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
    frame object — process_biomechanical_math needs it at full resolution separately, and
    this function's return value is never assigned back over the caller's `frame`."""
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= settings.VISION_IMAGE_MAX_DIMENSION:
        return frame
    scale = settings.VISION_IMAGE_MAX_DIMENSION / longest_side
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


async def _run_vision_triage(frame, session_id: str, discipline: str):
    """Classifies the tracking profile for a single BGR frame via GPT-4o-mini structured
    output, using the schema/prompt for the given discipline (BATTING or BOWLING) so the
    model can only pick a profile that's actually valid for that discipline."""
    schema = _TRIAGE_SCHEMAS_BY_DISCIPLINE[discipline]
    system_prompt = _TRIAGE_SYSTEM_PROMPTS_BY_DISCIPLINE[discipline]

    vision_frame = _downscale_for_vision(frame)
    ok, buffer = cv2.imencode(".jpg", vision_frame)
    if not ok:
        raise VisionTriageError("Failed to JPEG-encode the target frame.")
    base64_image = base64.b64encode(buffer).decode("utf-8")

    try:
        response = await openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            response_format=schema,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract technical error profile for MediaPipe routing."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ],
            timeout=12.0,
        )
    except Exception as exc:
        raise VisionTriageError(f"Vision triage API call failed: {exc}") from exc

    parsed = response.choices[0].message.parsed
    if parsed is None:
        refusal = response.choices[0].message.refusal
        raise VisionTriageError(f"Vision triage returned no parsed result (refusal={refusal!r}).")

    logger.info("Vision triage classified profile | session_id=%s profile=%s", session_id, parsed.assigned_profile)
    return parsed


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


async def _generate_video_insight(
    assigned_profile: str,
    primary_flaw: str,
    secondary_issues: List[str],
    measured_metric: float,
    session_id: str,
) -> tuple:
    """Returns (coach_summary, player_friendly_summary). Falls back to grounded, jargon-free
    templates — built from the real triage/pose output, never invented text — if the LLM call
    fails or returns nothing usable."""
    friendly_name = TRACKING_PROFILE_FRIENDLY_NAMES.get(assigned_profile, assigned_profile)
    fallback_coach = f"Flaw detected: {primary_flaw} ({abs(measured_metric):.2f}% deviation)."
    fallback_player = f"Let's work on your {friendly_name} — talk it through with your coach."

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
        metric_line = f"Metric: {assigned_profile} ({friendly_name}). Measured deviation: {measured_metric:.2f}%."
        player_instruction = "Speak directly to the player about focusing on standard form for this metric."

    try:
        response = await openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            response_format=VideoInsightSchema,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "A cricket coaching dashboard needs two summaries of the same flaw finding.\n"
                        f"{metric_line} Primary flaw: '{primary_flaw}'. Secondary issues: "
                        f"{secondary_issues}.\n"
                        "1) coach_summary: one precise sentence for a coach, referencing the metric and "
                        "the deviation. IMPORTANT: If this is a directional metric, you MUST use the literal "
                        "token '{direction}' in your sentence where the direction (e.g. high/low/left/right) "
                        "belongs (e.g., 'indicating the bowling arm is {direction} at release'). Do NOT write "
                        "the direction word yourself.\n"
                        "2) player_friendly_summary: one encouraging sentence a young player or their "
                        "parent can understand with no jargon, profile codes, or percentages — just what "
                        "changed and what to work on. "
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

        if assigned_profile in DIRECTIONAL_PROFILES:
            if "{direction}" in coach_summary:
                coach_summary = coach_summary.replace("{direction}", direction_label)
            else:
                coach_summary = f"{coach_summary.rstrip('.')} ({direction_label})."

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
        logger.error("OpenAI summary call failed | session_id=%s error=%s", session_id, exc)
        return fallback_coach, fallback_player


async def _process_event(
    event_index: int,
    frame_index: int,
    frame,
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
    the other concurrently-running events' results via asyncio.gather."""
    async with _event_semaphore:
        target_event_seconds = frame_index / fps

        try:
            triage = await _run_vision_triage(frame, session_id, discipline)
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

        measured_metric = process_biomechanical_math(frame, assigned_profile)
        if measured_metric is None:
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

        # Check if player has a reference baseline for this profile to detect regressions automatically
        regression_detected = False
        baseline_info = None

        try:
            # Query the baseline collection for this player and tracking profile
            baselines = baseline_collection.get(
                where={"$and": [{"player_id": player_id}, {"tracking_profile": assigned_profile}]}
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
            logger.error("Failed to check player baseline regression | player_id=%s profile=%s error=%s", player_id, assigned_profile, e)

        ai_insight, player_friendly_summary = await _generate_video_insight(
            assigned_profile, primary_flaw, secondary_issues, measured_metric, session_id
        )

        if regression_detected and baseline_info:
            deviation_pct = baseline_info["deviation_percentage"]
            issue_name = baseline_info["issue_name"]
            resolved_suffix = f" (marked fixed on {baseline_info['resolved_at']})" if baseline_info['resolved_at'] else " (previously marked fixed)"

            # Format specifications warnings
            ai_insight = f"AI Alert: Mechanical Regression Detected. Player is deviating from the Fixed Reference Video baseline by {deviation_pct:.1f}%. Habit returning: {issue_name}. {ai_insight}"
            player_friendly_summary = f"Your old habit of '{issue_name}'{resolved_suffix} is starting to show up a bit. Let's work to get it back to baseline! {player_friendly_summary}"

        shot_telemetry = {
            "session_id": session_id,
            "player_id": player_id,
            "event_index": event_index,
            "total_events": total_events,
            "discipline": discipline,
            "absolute_seconds": round(target_event_seconds, 2),
            "tracking_profile": assigned_profile,
            "measured_metric": round(measured_metric, 2),
            "primary_flaw": primary_flaw,
            "secondary_issues": secondary_issues,
            "ai_insight_summary": ai_insight,
            "player_friendly_summary": player_friendly_summary,
        }

        if regression_detected and baseline_info:
            shot_telemetry["regression_detected"] = True
            shot_telemetry["regression_info"] = baseline_info

        await redis_client.publish("shot_updates", json.dumps(shot_telemetry))
        logger.info(
            "Published shot telemetry | session_id=%s event=%s/%s",
            session_id,
            event_index + 1,
            total_events,
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
       triage picks which discipline-appropriate profile to track, MediaPipe pose math
       measures it, and a dual coach/player summary is generated.
    4. Publish one Redis message per event for the Node.js core backend to consume.

    Because events run concurrently, messages can arrive out of event_index order —
    consumers must key off event_index, not arrival order.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video | session_id=%s path=%s", session_id, video_path)
        cap.release()
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    events = find_motion_event_frames(video_path, fps)
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
            refined = find_bowling_release_frame(video_path, frame_index, fps)
            refined_events.append(refined if refined is not None else (frame_index, frame))
        events = refined_events

    await asyncio.gather(*(
        _process_event(event_index, frame_index, frame, fps, session_id, player_id, total_events, discipline)
        for event_index, (frame_index, frame) in enumerate(events)
    ))
