import requests
import logging

logger = logging.getLogger("osm")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_traffic_signals():
    query = """
[out:json];
node["highway"](6.947,79.853,6.965,79.875);
out;
"""

    logger.info("🌍 Querying Overpass API for Colombo 13 traffic signals")

    try:
        res = requests.post(
            OVERPASS_URL,
            data=query,
            timeout=30
        )
        res.raise_for_status()

        data = res.json()
        elements = data.get("elements", [])

        logger.info(f"🚦 Found {len(elements)} traffic signals")

        return [
            {
                "lat": el["lat"],
                "lng": el["lon"]
            }
            for el in elements
        ]

    except requests.exceptions.RequestException as e:
        logger.warning("⚠️ Overpass API failed (timeout / rate limit)")
        logger.debug(e)
        return []
