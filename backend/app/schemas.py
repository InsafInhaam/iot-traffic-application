from pydantic import BaseModel
from typing import Optional, Dict, Any

class IngestPayload(BaseModel):
    intersection: str
    timestamp: Optional[str] = None
    lanes: Dict[str, Dict]  # e.g. {"Lane_1": {"queue_m": 12.0, "count": 3}, ...}
    meta: Optional[Dict[str, Any]] = {}

class IntersectionOut(BaseModel):
    id: int
    name: str
    meta: Optional[Dict] = None

    class Config:
        orm_mode = True
