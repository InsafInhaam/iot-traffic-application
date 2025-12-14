import logging
import requests

logger = logging.getLogger("osm")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

_cached_signals = None


def fetch_traffic_signals(bbox):
    global _cached_signals

    if _cached_signals is not None:
        logger.info("🧠 Using cached OSM traffic signals")
        return _cached_signals

    logger.info("🌍 Querying Overpass API")

    try:
        res = requests.post(OVERPASS_URL, data=query, timeout=20)
        res.raise_for_status()
        data = res.json()

        _cached_signals = [
            {"lat": el["lat"], "lng": el["lon"]}
            for el in data.get("elements", [])
        ]

        logger.info(f"📍 Retrieved {_cached_signals.__len__()} signals")
        return _cached_signals

    except Exception:
        logger.warning("⚠️ Overpass API unavailable — returning empty list")
        _cached_signals = []
        return []
