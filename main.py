from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.errors import register_exception_handlers
from api.routes import router
from core.logging import configure_logging, get_logger
from services.broker import broker

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not broker.is_worker_process:
        await broker.startup()
        logger.info("Taskiq broker started")

    yield

    if not broker.is_worker_process:
        await broker.shutdown()
        logger.info("Taskiq broker stopped")


app = FastAPI(title="Super30 AI Engine", lifespan=lifespan)

register_exception_handlers(app)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
