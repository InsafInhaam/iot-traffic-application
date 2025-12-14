import asyncio
import json
from fastapi import WebSocket
from typing import List
from redis import asyncio as redis
from .config import settings

# simple websocket manager
redis_client = redis.from_url(
    "redis://redis:6379",
    decode_responses=True
)


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        to_send = json.dumps(message, default=str)
        for ws in list(self.active):
            try:
                await ws.send_text(to_send)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()

# Redis pub/sub broadcaster (background task)


async def redis_subscriber(app):
    # redis = aioredis.from_url(settings.REDIS_URL)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("traffic:events")
    async for msg in pubsub.listen():
        if msg is None:
            continue
        if msg.get("type") != "message":
            continue
        data = msg.get("data")
        try:
            payload = json.loads(data)
        except Exception:
            payload = {"raw": str(data)}
        await manager.broadcast(payload)


async def publish_event_to_redis(payload: dict):
    # redis = aioredis.from_url(settings.REDIS_URL)
    await redis_client.publish("traffic:events", json.dumps(payload, default=str))
