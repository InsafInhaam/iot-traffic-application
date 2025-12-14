import threading
import time
import random
from datetime import datetime

from app.services.osm import fetch_traffic_signals
from app.utils.geo import haversine
import logging
logger = logging.getLogger("simulator")

# -----------------------------------
# INITIAL APPROXIMATE COORDINATES
# -----------------------------------
INTERSECTION_COORDS = {
    "Intersection_A": {"lat": 6.943470, "lng": 79.864326},
    "Intersection_B": {"lat": 6.943220, "lng": 79.864216},
    "Intersection_C": {"lat": 6.943369, "lng": 79.864109},
    "Intersection_D": {"lat": 6.943523, "lng": 79.864241},
    "Intersection_E": {"lat": 6.948221, "lng": 79.858774},
}

INTERSECTIONS = {
    "Intersection_A": ["Intersection_B", "Intersection_C"],
    "Intersection_B": ["Intersection_A", "Intersection_D"],
    "Intersection_C": ["Intersection_A", "Intersection_D"],
    "Intersection_D": ["Intersection_B", "Intersection_C", "Intersection_E"],
    "Intersection_E": ["Intersection_D"],
}

state = {}

# Colombo bounding box
BBOX = (6.94, 79.85, 6.96, 79.87)

_simulator_started = False
_osm_snapped = False


# -----------------------------------
# SIGNAL LOGIC
# -----------------------------------
def compute_signal(vehicle_count: int):
    if vehicle_count <= 5:
        return "GREEN"
    elif vehicle_count <= 12:
        return "YELLOW"
    return "RED"


# -----------------------------------
# SNAP TO REAL OSM SIGNALS (RUN ONCE)
# -----------------------------------
def snap_intersections_to_osm():
    global _osm_snapped

    if _osm_snapped:
        logger.info("🔁 OSM snapping already done — skipping")
        return

    logger.info("🔗 Snapping intersections to OSM traffic signals")

    try:
        signals = fetch_traffic_signals(BBOX)
        logger.info(f"📡 Found {len(signals)} OSM traffic signals")
    except Exception:
        logger.warning("⚠️ OSM fetch failed — using static coordinates")
        _osm_snapped = True
        return

    if not signals:
        logger.warning("⚠️ No OSM signals found — using static coordinates")
        _osm_snapped = True
        return

    for name, coord in INTERSECTION_COORDS.items():
        best, best_dist = None, float("inf")

        for s in signals:
            d = haversine(coord["lat"], coord["lng"], s["lat"], s["lng"])
            if d < best_dist:
                best_dist = d
                best = s

        if best:
            INTERSECTION_COORDS[name]["lat"] = best["lat"]
            INTERSECTION_COORDS[name]["lng"] = best["lng"]
            logger.info(f"✅ {name} snapped ({round(best_dist,1)} m)")

    _osm_snapped = True
    logger.info("✅ OSM snapping finalized")

# -----------------------------------
# SIMULATOR LOOP
# -----------------------------------


def simulator_loop():
    logger.info("🚦 Simulator loop started")

    while True:
        for name, connections in INTERSECTIONS.items():
            vehicles = random.randint(0, 20)
            queue_m = round(vehicles * random.uniform(1.5, 3.0), 1)
            coords = INTERSECTION_COORDS[name]

            state[name] = {
                "id": name,
                "vehicles": vehicles,
                "queue_m": queue_m,
                "signal": compute_signal(vehicles),
                "last_update": datetime.now().strftime("%H:%M:%S"),
                "connections": connections,
                "lat": coords["lat"],
                "lng": coords["lng"],
            }

        logger.debug("🔄 Simulator tick")
        time.sleep(2)


# -----------------------------------
# PUBLIC API
# -----------------------------------
def start_simulator():
    global _simulator_started

    if _simulator_started:
        logger.info("🔁 Simulator already running — skipping")
        return

    logger.info("▶ Starting simulator thread")
    t = threading.Thread(target=simulator_loop, daemon=True)
    t.start()
    _simulator_started = True


def get_network_state():
    return list(state.values())
