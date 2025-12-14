from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from . import db, models, schemas, crud, events
from .db import SessionLocal, init_db
from app.tasks import simulator
from app.services.osm import fetch_traffic_signals

import logging
import logging.config
import sys
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": LOG_FILE,
            "maxBytes": 5 * 1024 * 1024,  # 5 MB
            "backupCount": 3,
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
    "loggers": {
        "simulator": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "osm": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
})

logger = logging.getLogger("simulator")

app = FastAPI(title="Traffic Multi-Intersection API")

# Initialize DB
init_db()


def get_db():
    dbs = SessionLocal()
    try:
        yield dbs
    finally:
        dbs.close()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ SINGLE startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🔥 Application startup event reached")

    try:
        simulator.snap_intersections_to_osm()
        logger.info("✅ OSM snapping completed")
    except Exception as e:
        logger.exception("❌ OSM snapping failed")

    simulator.start_simulator()
    logger.info("🚦 Simulator started")

    loop = asyncio.get_event_loop()
    loop.create_task(events.redis_subscriber(app))


@app.post("/api/v1/ingest", status_code=201)
async def ingest(payload: schemas.IngestPayload, background: BackgroundTasks, db: Session = Depends(get_db)):
    inter = crud.get_or_create_intersection(
        db, payload.intersection, meta=payload.meta)
    meas = crud.create_measurements(db, inter.id, payload.lanes)
    crud.create_event(db, inter.id, "ingest", payload.dict())

    background.add_task(events.publish_event_to_redis, {
        "type": "ingest",
        "intersection": inter.name,
        "payload": payload.dict()
    })

    return {"ok": True, "intersection": inter.name, "stored": len(meas)}


@app.get("/api/v1/events/latest")
async def get_latest_events():
    try:
        return {"events": simulator.get_live_events() or []}
    except Exception as e:
        print("events error:", e)
        return {"events": []}


@app.get("/api/v1/metrics/live")
async def get_live_metrics():
    try:
        return {"points": simulator.get_chart_data() or []}
    except Exception as e:
        print("metrics error:", e)
        return {"points": []}


@app.get("/api/v1/network")
async def get_intersection_network():
    return {"intersections": simulator.get_network_state()}


@app.get("/api/v1/osm/traffic-signals")
async def get_osm_traffic_signals():
    # Colombo bounding box (adjust later)
    bbox = (6.94, 79.85, 6.96, 79.86)

    try:
        signals = fetch_traffic_signals(bbox)
        return {"signals": signals}
    except Exception as e:
        print("OSM error:", e)
        return {"signals": []}
