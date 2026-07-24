from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    REDIS_URL: str = "redis://localhost:6379"
    OPENAI_API_KEY: str = Field(min_length=1)
    CHROMA_DB_PATH: str = "./chroma_data"
    POSE_LANDMARKER_MODEL_PATH: str = "./models/pose_landmarker_lite.task"

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
    # concurrently. Tested at 4 against a real rate-limited account and it made
    # recovery WORSE, not better: the OpenAI SDK doesn't jitter its wait when honoring
    # the server's explicit Retry-After header, so concurrent requests told "retry in
    # 250ms" collide in lockstep, and concurrency front-loads more token demand into a
    # shorter window against an already-saturated ceiling. Default to 1 (sequential)
    # until there's headroom evidence that raising it actually helps.
    VIDEO_EVENT_CONCURRENCY: int = 1
    # Vision-model image tokenization cost scales with resolution (a full 1280x720 frame
    # costs meaningfully more tokens than a downscaled one). This only applies to the copy
    # sent to the vision API for triage classification — MediaPipe pose detection always
    # gets the original, untouched full-resolution frame separately.
    VISION_IMAGE_MAX_DIMENSION: int = 640


settings = Settings()
