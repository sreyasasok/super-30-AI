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
