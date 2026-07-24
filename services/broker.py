import redis.asyncio as aioredis
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend

from config import settings

result_backend = RedisAsyncResultBackend(redis_url=settings.REDIS_URL)
broker = ListQueueBroker(url=settings.REDIS_URL).with_result_backend(result_backend)

redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
