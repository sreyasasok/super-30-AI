import os
import tempfile
from urllib.parse import urlparse

import httpx

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1MB streaming chunks, so a large video never sits fully in memory.


class VideoFetchError(Exception):
    """Raised when a video source that looks like a URL can't be downloaded. Callers turn
    this into a clean 4xx (sync endpoints) or a logged early-return (the async pipeline,
    which has no synchronous caller left to hand an HTTP error to)."""


def is_remote_url(video_source: str) -> bool:
    """True for http(s) sources (e.g. a public S3/GCS/CDN link); false for anything else,
    which is then treated as a local filesystem path exactly like before this existed."""
    return urlparse(video_source).scheme in ("http", "https")


async def resolve_video_source(video_source: str) -> tuple[str, bool]:
    """Returns (local_path, is_temp_file).

    If `video_source` is an http(s) URL, streams it to a temp file on disk and returns that
    path with is_temp_file=True — the caller MUST pass this to cleanup_video_source when done
    to avoid leaking temp files. If it's already a local path, returned unchanged with
    is_temp_file=False and there is nothing to clean up.
    """
    if not is_remote_url(video_source):
        return video_source, False

    suffix = os.path.splitext(urlparse(video_source).path)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        async with httpx.AsyncClient(timeout=settings.VIDEO_DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
            async with client.stream("GET", video_source) as response:
                if response.status_code != 200:
                    raise VideoFetchError(
                        f"Failed to download video (HTTP {response.status_code}): {video_source}"
                    )
                with open(tmp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                        f.write(chunk)
    except httpx.HTTPError as exc:
        os.remove(tmp_path)
        raise VideoFetchError(f"Failed to download video from URL: {video_source} ({exc})") from exc
    except VideoFetchError:
        os.remove(tmp_path)
        raise
    except Exception:
        os.remove(tmp_path)
        raise

    logger.info("Downloaded remote video | url=%s local_path=%s", video_source, tmp_path)
    return tmp_path, True


def cleanup_video_source(local_path: str, is_temp_file: bool) -> None:
    """No-op for a real local path (is_temp_file=False) — never deletes a caller-owned file.
    Only removes files this module itself downloaded."""
    if not is_temp_file:
        return
    try:
        if os.path.isfile(local_path):
            os.remove(local_path)
    except OSError as exc:
        logger.warning("Failed to clean up downloaded temp video | path=%s error=%s", local_path, exc)
