import redis.asyncio as aioredis
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend

from config import settings

result_backend = RedisAsyncResultBackend(redis_url=settings.REDIS_URL)
# socket_timeout=None: redis-py 8.x defaults every connection's socket_timeout to 5 seconds
# (redis.asyncio.connection.DEFAULT_SOCKET_TIMEOUT), which is fine for quick request/response
# commands but fatal here — ListQueueBroker's worker loop sits in a long-blocking BRPOP
# waiting for the next job, and an idle queue for more than 5s (the normal case) makes the
# client kill its own socket and raise TimeoutError, crashing the whole worker process
# (including any job it's concurrently executing) in a restart loop. None disables that
# client-side read timeout so BRPOP can block for as long as it's actually meant to.
# socket_connect_timeout is untouched (stays at the library default) — a hang on the initial
# TCP handshake should still fail fast.
broker = ListQueueBroker(url=settings.REDIS_URL, socket_timeout=None).with_result_backend(result_backend)

redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

# Registers @broker.task-decorated functions on `broker` as a side effect of this import.
# Taskiq's worker CLI only knows about tasks from modules it's explicitly told to import
# (either via a trailing `taskiq worker services.broker:broker tasks.video_tasks` CLI arg, or
# by that module being imported somewhere reachable from the broker module itself, as here).
# Without this, a worker started as bare `taskiq worker services.broker:broker` dequeues a job
# fine but can't find the code for it: "task ... is not found. Maybe you forgot to import it?"
# Placed at the bottom, after `broker`/`redis_client` are defined, since tasks.video_tasks
# imports them back — importing any earlier would be a circular import.
import tasks.video_tasks  # noqa: F401
