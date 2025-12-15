from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import asyncio
import logging
import logging.config
import os
import sys

from .db import SessionLocal, init_db
from . import schemas, crud, events
from app.tasks import simulator

# -----------------------------
# LOGGING SETUP
# -----------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": LOG_FILE,
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
})

logger = logging.getLogger("app")

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="Traffic Multi-Intersection API")

init_db()

# -----------------------------
# DB DEPENDENCY
# -----------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# STARTUP
# -----------------------------


@app.on_event("startup")
async def startup_event():
    logger.info("🔥 Application startup")
    simulator.start_simulator()
    loop = asyncio.get_event_loop()
    loop.create_task(events.redis_subscriber(app))


# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.get("/api/v1/network")
async def network():
    return {"intersections": simulator.get_network_state()}


@app.get("/api/v1/events/latest")
async def latest_events():
    return {"events": simulator.get_live_events()}


@app.get("/api/v1/metrics/live")
async def live_metrics():
    return {"points": simulator.get_chart_data()}


@app.post("/api/v1/ingest")
async def ingest(payload: schemas.IngestPayload, db: Session = Depends(get_db)):
    inter = crud.get_or_create_intersection(
        db, payload.intersection, meta=payload.meta
    )

    crud.create_measurements(db, inter.id, payload.lanes)
    crud.create_event(db, inter.id, "ingest", payload.dict())

    return {"ok": True}
