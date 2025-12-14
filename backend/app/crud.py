from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime

def get_or_create_intersection(db: Session, name: str, meta=None):
    q = db.query(models.Intersection).filter(models.Intersection.name == name).first()
    if q:
        return q
    i = models.Intersection(name=name, meta=meta or {})
    db.add(i)
    db.commit()
    db.refresh(i)
    return i

def create_measurements(db: Session, intersection_id: int, lanes: dict):
    out = []
    for lane, data in lanes.items():
        m = models.Measurement(
            intersection_id=intersection_id,
            lane=lane,
            queue_m=float(data.get("queue_m", 0.0)),
            vehicle_count=int(data.get("count", data.get("vehicle_count", 0))),
            ts=datetime.utcnow()
        )
        db.add(m)
        out.append(m)
    db.commit()
    for m in out:
        db.refresh(m)
    return out

def create_event(db: Session, intersection_id: int, etype: str, payload: dict):
    ev = models.Event(intersection_id=intersection_id, type=etype, payload=payload)
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev
