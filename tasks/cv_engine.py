import os
from dataclasses import dataclass

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from scipy.signal import find_peaks
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# Landmark indices from the BlazePose 33-point topology.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_HIP, RIGHT_HIP = 23, 24
RIGHT_KNEE = 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# The 10 named joints the profile formulas below actually read. Deliberately not the full
# 33-point BlazePose topology — the other 23 points feed no metric, so there's nothing for
# a coach to usefully drag when manually correcting a pose (see /pose-corrections).
LANDMARK_NAME_TO_INDEX = {
    "NOSE": NOSE,
    "LEFT_SHOULDER": LEFT_SHOULDER,
    "RIGHT_SHOULDER": RIGHT_SHOULDER,
    "RIGHT_ELBOW": RIGHT_ELBOW,
    "RIGHT_WRIST": RIGHT_WRIST,
    "LEFT_HIP": LEFT_HIP,
    "RIGHT_HIP": RIGHT_HIP,
    "RIGHT_KNEE": RIGHT_KNEE,
    "LEFT_ANKLE": LEFT_ANKLE,
    "RIGHT_ANKLE": RIGHT_ANKLE,
}

# Which of the named joints each tracking profile's formula reads — read directly off the
# `if profile == ...` bodies in compute_metric_from_landmarks below. Lets the coach app ask
# for only the joints relevant to the flaw it's showing, instead of all 10 every time.
PROFILE_RELEVANT_LANDMARKS = {
    # Batting
    "SHOULDER_TILT": ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
    "HAND_BACKLIFT": ["RIGHT_WRIST", "RIGHT_SHOULDER"],
    "FOOTWORK_WIDTH": ["LEFT_ANKLE", "RIGHT_ANKLE"],
    "FRONT_KNEE_BEND": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
    "HEAD_STABILITY": ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
    "ELBOW_ELEVATION": ["RIGHT_ELBOW", "RIGHT_SHOULDER"],
    # Bowling
    "BOWLING_ARM_HEIGHT": ["RIGHT_WRIST", "RIGHT_SHOULDER"],
    "FRONT_KNEE_BRACE": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
    "RELEASE_ALIGNMENT": ["RIGHT_WRIST", "RIGHT_SHOULDER"],
}


@dataclass
class Point:
    """Minimal x/y landmark stand-in. MediaPipe's own landmark objects aren't user-
    constructible, but compute_metric_from_landmarks only ever reads `.x`/`.y`, so this is
    enough to splice a coach's manual correction into an otherwise-real landmark list."""

    x: float
    y: float

# Profiles where the SIGN of the measurement carries real coaching meaning (e.g. "arm too
# low" vs "arm too high" are different, distinguishable flaws) — these return a signed
# value from process_biomechanical_math instead of an absolute distance. Confirmed via a
# real-footage audit that unsigned values were actively misleading: a clearly high, well-
# extended bowling arm produced the same large |value| a genuinely low arm would, and the
# LLM's text claim ("too low") directly contradicted what the frame showed.
DIRECTIONAL_PROFILES = {"HAND_BACKLIFT", "BOWLING_ARM_HEIGHT", "ELBOW_ELEVATION", "RELEASE_ALIGNMENT"}


def get_directional_label(profile: str, val: float) -> str:
    """Computes a precise, jargon-free direction label for signed values in DIRECTIONAL_PROFILES.
    Using code to determine the direction is far more reliable than expecting LLMs to consistently
    evaluate positive vs. negative sign conventions during structured text generation."""
    if profile not in DIRECTIONAL_PROFILES:
        return ""
    if profile in ("HAND_BACKLIFT", "BOWLING_ARM_HEIGHT"):
        # Image coordinates go downward, so positive means wrist is BELOW the shoulder midpoint.
        return "too low" if val >= 0 else "too high"
    if profile == "ELBOW_ELEVATION":
        return "too low" if val >= 0 else "too high"
    if profile == "RELEASE_ALIGNMENT":
        # Positive means wrist is to the right of the shoulder midpoint.
        return "too far right" if val >= 0 else "too far left"
    return ""


def get_player_adjustment_label(profile: str, val: float) -> str:
    """Computes a precise physical correction verb (e.g. raise, lower) corresponding to the
    signed direction of a metric. Handing this directly to the model completely prevents the
    model's semantic biases from generating contradictory instructions for young players."""
    label = get_directional_label(profile, val)
    if not label:
        return ""
    if label == "too low":
        return "raising"
    if label == "too high":
        return "lowering"
    if label == "too far right":
        return "moving left"
    if label == "too far left":
        return "moving right"
    return ""


TRACKING_PROFILE_FRIENDLY_NAMES = {
    # Batting
    "FOOTWORK_WIDTH": "foot positioning and stance width",
    "HAND_BACKLIFT": "hand and bat lift height",
    "SHOULDER_TILT": "shoulder balance and levelness",
    "FRONT_KNEE_BEND": "front knee bend through the shot",
    "HEAD_STABILITY": "head position and stillness",
    "ELBOW_ELEVATION": "elbow height during the shot",
    # Bowling
    "BOWLING_ARM_HEIGHT": "bowling arm height at release",
    "FRONT_KNEE_BRACE": "front leg brace on landing",
    "RELEASE_ALIGNMENT": "arm alignment through release",
}

if not os.path.isfile(settings.POSE_LANDMARKER_MODEL_PATH):
    raise FileNotFoundError(
        f"Pose landmarker model not found at '{settings.POSE_LANDMARKER_MODEL_PATH}'. "
        "Run scripts/download_models.sh first."
    )

_landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=settings.POSE_LANDMARKER_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
)


def calculate_joint_angle(a, b, c) -> float:
    """Angle at vertex b formed by rays b->a and b->c, in degrees."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / norm
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))


MOTION_DIFF_THRESHOLD = 25


def find_peak_motion_frame(cap: cv2.VideoCapture) -> tuple[int, np.ndarray] | None:
    """Scans the whole clip via adjacent-frame differencing and returns (frame_index, frame)
    for the moment with the most changed pixels. Keeps frames in memory during the single
    forward scan rather than re-seeking afterward, which avoids a second full read pass and
    the frame-seek imprecision some codecs have on non-keyframe positions."""
    ret, frame1 = cap.read()
    if not ret:
        return None

    peak_index = 0
    peak_frame = frame1
    max_moving_pixels = 0

    frame_idx = 0
    ret, frame2 = cap.read()
    while ret:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        moving_pixels = cv2.countNonZero(thresh)

        if moving_pixels > max_moving_pixels:
            max_moving_pixels = moving_pixels
            peak_index = frame_idx + 1
            peak_frame = frame2

        frame1 = frame2
        frame_idx += 1
        ret, frame2 = cap.read()

    return peak_index, peak_frame


# Minimum real-world seconds between two distinct motion events (deliveries), so one
# delivery's backswing-impact-followthrough doesn't get split into multiple false events.
MIN_EVENT_SPACING_SECONDS = 2.5

# Real edit cuts between separate clips in a compilation video produce a FAR larger
# frame-to-frame difference than any genuine in-scene motion — confirmed empirically on
# real footage: cuts measured 41,777-311,099 (4.5%-33.8% of frame) vs a confirmed real
# swing's max of 31,318 (3.4%). Naively using mean+std as the noise floor doesn't work
# because a handful of huge cut outliers inflate BOTH the mean and std enough that the
# floor ends up ABOVE genuine motion and only cut spikes clear it — which is exactly what
# happened: the very first version of this function detected 18 "events" that were all
# edit cuts (repeated near-identical stance frames), not real swings. Median and MAD
# (median absolute deviation) are used instead because they aren't dragged around by a
# small number of extreme outliers the way mean/std are.
#
# These multipliers are empirically calibrated on one real test video, not universally
# derived — same caveat as DRILL_SIMILARITY_CUTOFF before it was measured against real
# embeddings. Expect to retune as more real session footage becomes available.
NOISE_FLOOR_MAD_MULTIPLIER = 1.0
CUT_EXCLUSION_MAD_MULTIPLIER = 2.5


def _compute_motion_signal(video_path: str) -> np.ndarray:
    """Sequential pass computing the moving-pixel count between each adjacent frame pair.
    Does NOT hold frames in memory — a multi-minute session video holding every raw BGR
    frame would be a real memory problem, unlike the short clips `find_peak_motion_frame`
    was built for."""
    cap = cv2.VideoCapture(video_path)
    signal: list[int] = []

    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        return np.array([])

    ret, frame2 = cap.read()
    while ret:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        signal.append(cv2.countNonZero(thresh))

        frame1 = frame2
        ret, frame2 = cap.read()

    cap.release()
    return np.array(signal, dtype=np.float64)


def find_motion_event_frames(video_path: str, fps: float) -> list[tuple[int, np.ndarray]]:
    """Detects every distinct motion event in a video (e.g. each delivery in a full net
    session), not just the single global peak `find_peak_motion_frame` finds. Returns a
    list of (frame_index, frame) sorted by frame index.

    Takes a video_path rather than an open cv2.VideoCapture (unlike find_peak_motion_frame)
    because this needs two full passes: one lightweight pass over the motion-signal array
    only, then a second sequential re-read (not cap.set() seeking, for the same seek-
    imprecision reason documented on find_peak_motion_frame) to extract just the frames at
    the detected event indices."""
    signal = _compute_motion_signal(video_path)
    if signal.size == 0:
        return []

    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median)))
    noise_floor = median + NOISE_FLOOR_MAD_MULTIPLIER * mad
    cut_ceiling = median + CUT_EXCLUSION_MAD_MULTIPLIER * mad
    min_distance = max(1, int(MIN_EVENT_SPACING_SECONDS * fps))

    peak_indices, _ = find_peaks(signal, height=(noise_floor, cut_ceiling), distance=min_distance)
    if peak_indices.size == 0:
        return []

    # +1: signal[i] is the motion between frame i and frame i+1, so the event frame is i+1 —
    # same convention find_peak_motion_frame uses (peak_index = frame_idx + 1).
    target_frame_indices = {int(i) + 1 for i in peak_indices}

    cap = cv2.VideoCapture(video_path)
    events: list[tuple[int, np.ndarray]] = []
    frame_idx = 0
    ret, frame = cap.read()
    while ret:
        if frame_idx in target_frame_indices:
            events.append((frame_idx, frame))
        frame_idx += 1
        ret, frame = cap.read()
    cap.release()

    events.sort(key=lambda event: event[0])
    return events


def detect_landmarks(frame: np.ndarray):
    """Runs MediaPipe pose detection on a single BGR frame. Returns the raw 33-point
    landmark list, or None if no pose could be detected."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
    result = _landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


def extract_named_landmarks(frame: np.ndarray) -> dict[str, dict[str, float]] | None:
    """Runs pose detection and returns just the 10 named joints (see
    LANDMARK_NAME_TO_INDEX) as {name: {"x": ..., "y": ...}}, or None if no pose was
    detected. Used by /frames/pose-landmarks to hand the coach app something to render and
    let a coach drag into a corrected position."""
    landmarks = detect_landmarks(frame)
    if landmarks is None:
        return None
    return {
        name: {"x": landmarks[idx].x, "y": landmarks[idx].y}
        for name, idx in LANDMARK_NAME_TO_INDEX.items()
    }


def apply_landmark_corrections(landmarks, corrections: dict[str, tuple[float, float]]) -> list:
    """Returns a copy of `landmarks` (as a plain list) with the named joints in
    `corrections` overridden to Point(x, y); every other joint is left as the original
    MediaPipe landmark object. Used by /pose-corrections to compute what a tracking
    profile's metric would be under a coach's manual joint correction."""
    corrected = list(landmarks)
    for name, (x, y) in corrections.items():
        corrected[LANDMARK_NAME_TO_INDEX[name]] = Point(x=x, y=y)
    return corrected


def compute_metric_from_landmarks(landmarks, profile: str) -> float | None:
    """Pure math: given a landmark list (real MediaPipe output, or a copy with specific
    joints overridden by a coach's manual correction — see /pose-corrections), returns the
    metric for `profile`. Only ever reads `.x`/`.y` off each landmark, so it works
    identically on either input. Returns None only for an unrecognized profile string."""
    # --- Batting ---
    if profile == "SHOULDER_TILT":
        return abs(landmarks[LEFT_SHOULDER].y - landmarks[RIGHT_SHOULDER].y) * 100

    if profile == "HAND_BACKLIFT":
        # Signed: image y increases downward, so wrist.y > shoulder.y means the hand sits
        # BELOW shoulder height. Positive = backlift too low; negative = backlift too high.
        return (landmarks[RIGHT_WRIST].y - landmarks[RIGHT_SHOULDER].y) * 100

    if profile == "FOOTWORK_WIDTH":
        return abs(landmarks[LEFT_ANKLE].x - landmarks[RIGHT_ANKLE].x) * 100

    if profile == "FRONT_KNEE_BEND":
        # Hip-knee-ankle angle: 180 degrees is a straight leg, so we express this as
        # deviation-from-straight (0% = straight, higher % = more bend) to stay on the
        # same 0-100-ish scale the rest of these metrics use, since downstream summary
        # text always reports "measured_metric%".
        angle = calculate_joint_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        return max(0.0, 100.0 - (angle / 180.0 * 100.0))

    if profile == "HEAD_STABILITY":
        shoulder_midpoint_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
        return abs(landmarks[NOSE].x - shoulder_midpoint_x) * 100

    if profile == "ELBOW_ELEVATION":
        # Signed, same convention as HAND_BACKLIFT: positive = elbow low, negative = high.
        return (landmarks[RIGHT_ELBOW].y - landmarks[RIGHT_SHOULDER].y) * 100

    # --- Bowling ---
    if profile == "BOWLING_ARM_HEIGHT":
        # Signed, same convention as HAND_BACKLIFT — this was the specific bug an audit
        # caught: the old abs() version reported the SAME large magnitude for a clearly
        # high, well-extended arm as it would for a genuinely low one, so the accompanying
        # LLM text ("arm too low") could contradict what the frame actually showed.
        return (landmarks[RIGHT_WRIST].y - landmarks[RIGHT_SHOULDER].y) * 100

    if profile == "FRONT_KNEE_BRACE":
        angle = calculate_joint_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        return max(0.0, 100.0 - (angle / 180.0 * 100.0))

    if profile == "RELEASE_ALIGNMENT":
        # Signed: positive = wrist right of shoulder, negative = left, at release.
        return (landmarks[RIGHT_WRIST].x - landmarks[RIGHT_SHOULDER].x) * 100

    logger.warning("Unknown tracking profile requested: %s", profile)
    return None


def process_biomechanical_math(frame: np.ndarray, profile: str) -> float | None:
    """Runs pose detection on a single BGR frame and returns the metric for `profile`, or
    None if no pose could be detected — callers must treat None as "couldn't measure this",
    not as a genuine zero-deviation reading. Conflating the two was a real bug: a frame
    where MediaPipe found no landmarks at all still got reported as measured_metric=0.0,
    indistinguishable from an actual perfect-technique measurement."""
    landmarks = detect_landmarks(frame)
    if landmarks is None:
        logger.warning("No pose landmarks detected for profile=%s", profile)
        return None
    return compute_metric_from_landmarks(landmarks, profile)


RELEASE_SEARCH_WINDOW_SECONDS = 0.75
# Sample every Nth frame within the search window rather than every frame, to bound
# MediaPipe cost — a full-rate scan of the window isn't needed to find the elevation peak.
RELEASE_SEARCH_STRIDE = 2


def find_bowling_release_frame(
    video_path: str, coarse_frame_index: int, fps: float
) -> tuple[int, np.ndarray] | None:
    """Refines a coarse motion-event frame to the actual release instant for a bowling
    action. Confirmed via a real-footage audit that raw frame-differencing can't tell
    run-up motion or follow-through from the release itself — both involve large
    whole-body movement, so the coarse event sometimes landed mid-run-up or in
    follow-through instead of release, and the reported flaw evaluated the wrong instant
    entirely. This scans a window around the coarse event and picks the frame where the
    bowling wrist sits highest above the shoulder — a real, distinguishing signature of
    release that plain pixel motion can't provide, since arms swing near torso height
    while running but the bowling arm is characteristically raised well above the
    shoulder at release."""
    window = int(RELEASE_SEARCH_WINDOW_SECONDS * fps)
    start = max(0, coarse_frame_index - window)
    end = coarse_frame_index + window

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    best_index = None
    best_frame = None
    best_elevation = None

    frame_idx = start
    ret, frame = cap.read()
    while ret and frame_idx <= end:
        if (frame_idx - start) % RELEASE_SEARCH_STRIDE == 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
            result = _landmarker.detect(mp_image)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                # Positive = wrist above shoulder (image y decreases upward).
                elevation = landmarks[RIGHT_SHOULDER].y - landmarks[RIGHT_WRIST].y
                if best_elevation is None or elevation > best_elevation:
                    best_elevation = elevation
                    best_index, best_frame = frame_idx, frame
        frame_idx += 1
        ret, frame = cap.read()
    cap.release()

    if best_frame is None:
        return None
    return best_index, best_frame
