from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    REDIS_URL: str = "redis://localhost:6379"
    OPENAI_API_KEY: str = Field(min_length=1)
    # Model for both the vision triage/classification calls and the structured-output insight
    # generation. Originally set to flagship gpt-5 (over gpt-4o-mini) for stronger instruction-
    # following, which measurably reduced a class of bug where the model ignored a conditional
    # prompt rule (e.g. emitting a direction token on a non-directional metric).
    # Re-tested gpt-5 vs gpt-5-mini specifically for directional accuracy (does the model
    # correctly judge "arm too high" vs "too low" from a still frame): with the profile held
    # fixed so it's a true apples-to-apples test, BOTH scored only 1/3 correct — statistically
    # indistinguishable, and neither is trustworthy for this without grounding in the actual
    # measurement (see the primary_flaw correction in tasks/video_tasks.py). Since that fix
    # makes model choice irrelevant to this specific correctness concern, and gpt-5-mini is
    # measurably faster on the realistic multimodal+structured-output triage call (~2.25s vs
    # ~6.49s per call in isolated testing) with no loss in descriptive quality (unlike
    # gpt-5-nano, which degraded to just echoing the profile name as the flaw description),
    # gpt-5-mini is the better default. Centralized here so it's one knob to tune.
    OPENAI_MODEL: str = "gpt-5-mini"
    # "minimal" keeps latency in line with the per-event timeouts below (10-12s each, and the
    # pipeline calls this multiple times per video) — these are short classification/summary
    # calls, not tasks that benefit from GPT-5's deeper step-by-step reasoning.
    OPENAI_REASONING_EFFORT: str = "minimal"
    CHROMA_DB_PATH: str = "./chroma_data"
    POSE_LANDMARKER_MODEL_PATH: str = "./models/pose_landmarker_lite.task"
    # video_path/current_video_path fields accept a public http(s) URL (e.g. an S3/GCS/CDN
    # link) in addition to a local filesystem path; this bounds how long we'll wait for that
    # download before giving up. Generous because coaching session videos can be sizeable.
    VIDEO_DOWNLOAD_TIMEOUT_SECONDS: float = 120.0

    REGRESSION_THRESHOLD: float = 0.15
    # Calibrated against real text-embedding-3-small cosine similarities on a 6-drill /
    # 3-category test set: separates all true negatives (max 0.4656) from true matches
    # (next lowest 0.4801). Retune as the real drill library grows.
    DRILL_SIMILARITY_CUTOFF: float = 0.47
    LOG_LEVEL: str = "INFO"

    # OpenAI's SDK already retries on 429/5xx with real exponential backoff honoring the
    # server's Retry-After header — this just raises the retry ceiling above the SDK
    # default (2), which isn't enough for a burst of many events' calls in one video.
    OPENAI_MAX_RETRIES: int = 5
    # How many motion events in one video get their triage+pose+insight pipeline run
    # concurrently. Originally defaulted to 1 (sequential) after testing 4 against a real
    # rate-limited account made retry recovery WORSE — the SDK's Retry-After backoff isn't
    # jittered, so concurrent requests told "retry in 250ms" collided in lockstep, and
    # concurrency front-loaded more token demand into an already-saturated ceiling.
    # Re-tested at 8 against real request headroom (499/500 requests, 499997/500000 tokens
    # remaining per response headers — nowhere near that earlier ceiling): cut a 19-event
    # video's processing time from ~123s to ~28-33s with zero retries or 429s across
    # multiple runs. Also confirmed the shared MediaPipe `_landmarker` instance is safe
    # under this concurrency — 20 truly-concurrent calls on identical input returned
    # bit-for-bit identical results every time, since its synchronous, un-awaited `.detect()`
    # call can never actually interleave with another task within one asyncio event loop.
    # Revisit downward only if a future account/tier has a tighter rate limit than this.
    VIDEO_EVENT_CONCURRENCY: int = 8
    # Vision-model image tokenization cost scales with resolution (a full 1280x720 frame
    # costs meaningfully more tokens than a downscaled one). This only applies to the copy
    # sent to the vision API for triage classification — MediaPipe pose detection always
    # gets the original, untouched full-resolution frame separately.
    VISION_IMAGE_MAX_DIMENSION: int = 640


settings = Settings()
