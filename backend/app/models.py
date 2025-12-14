from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from .db import Base
from sqlalchemy.orm import relationship

class Intersection(Base):
    __tablename__ = "intersections"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    meta = Column(JSON, nullable=True)  # geo coords, config

    events = relationship("Event", back_populates="intersection")
    measurements = relationship("Measurement", back_populates="intersection")

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    intersection_id = Column(Integer, ForeignKey("intersections.id"))
    type = Column(String)  # e.g., green_start, green_end, emergency, ingest
    payload = Column(JSON)
    ts = Column(DateTime(timezone=True), server_default=func.now())

    intersection = relationship("Intersection", back_populates="events")

class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True, index=True)
    intersection_id = Column(Integer, ForeignKey("intersections.id"))
    lane = Column(String, index=True)
    queue_m = Column(Float)
    vehicle_count = Column(Integer)
    ts = Column(DateTime(timezone=True), server_default=func.now())

    intersection = relationship("Intersection", back_populates="measurements")
