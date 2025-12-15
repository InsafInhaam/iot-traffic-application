import threading
import time
import random
from datetime import datetime
import logging

logger = logging.getLogger("simulator")

# -----------------------------
# INTERSECTION COORDINATES
# -----------------------------
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
_events = []
_chart_points = []

_simulator_started = False

# -----------------------------
# SIGNAL LOGIC
# -----------------------------
def compute_signal(vehicles: int):
    if vehicles <= 5:
        return "GREEN"
    elif vehicles <= 12:
        return "YELLOW"
    return "RED"


# -----------------------------
# METRICS + EVENTS
# -----------------------------
def record_event(event_type, intersection):
    _events.append({
        "type": event_type,
        "intersection": intersection,
        "time": datetime.now().strftime("%H:%M:%S"),
    })

    if len(_events) > 50:
        _events.pop(0)


def record_metrics(total_vehicles):
    _chart_points.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "vehicles": total_vehicles,
    })

    if len(_chart_points) > 30:
        _chart_points.pop(0)


# -----------------------------
# SIMULATOR LOOP
# -----------------------------
def simulator_loop():
    logger.info("🚦 Simulator loop started")

    while True:
        total_vehicles = 0

        for name, connections in INTERSECTIONS.items():
            vehicles = random.randint(0, 20)
            queue_m = round(vehicles * random.uniform(1.5, 3.0), 1)

            total_vehicles += vehicles

            state[name] = {
                "id": name,
                "vehicles": vehicles,
                "queue_m": queue_m,
                "signal": compute_signal(vehicles),
                "connections": connections,
                "lat": INTERSECTION_COORDS[name]["lat"],
                "lng": INTERSECTION_COORDS[name]["lng"],
                "last_update": datetime.now().strftime("%H:%M:%S"),
            }

            record_event("update", name)

        record_metrics(total_vehicles)
        time.sleep(2)


# -----------------------------
# PUBLIC API
# -----------------------------
def start_simulator():
    global _simulator_started

    if _simulator_started:
        logger.info("🔁 Simulator already running")
        return

    t = threading.Thread(target=simulator_loop, daemon=True)
    t.start()
    _simulator_started = True
    logger.info("▶ Simulator thread started")


def get_network_state():
    return list(state.values())


def get_live_events():
    return _events


def get_chart_data():
    return _chart_points
