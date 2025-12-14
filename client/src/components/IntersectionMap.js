"use client";

import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import * as L from "leaflet";
import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const lightIcon = (signal) =>
  L.divIcon({
    className: "custom-marker",
    html: `
      <div class="traffic-light">
        <span class="red ${signal === "RED" ? "on" : ""}"></span>
        <span class="yellow ${signal === "YELLOW" ? "on" : ""}"></span>
        <span class="green ${signal === "GREEN" ? "on" : ""}"></span>
      </div>
    `,
  });

export default function IntersectionMap({ network }) {
  const [osmSignals, setOsmSignals] = useState([]);

  useEffect(() => {
    fetch(API_BASE + "/api/v1/osm/traffic-signals")
      .then((r) => r.json())
      .then((d) => setOsmSignals(d.signals || []))
      .catch(() => setOsmSignals([]));
  }, []);

  return (
    <MapContainer
      center={[6.9486, 79.8572]}
      zoom={16}
      style={{ height: "500px", width: "100%" }}
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

      {/* REAL OSM TRAFFIC SIGNALS */}
      {osmSignals.map((s) => (
        <Marker key={s.id} position={[s.lat, s.lng]} icon={lightIcon(s.signal)}>
          <Popup>
            <strong>Traffic Signal</strong>
            <br />
            Source: OpenStreetMap
          </Popup>
        </Marker>
      ))}

      {/* SIMULATED INTERSECTIONS (optional overlay) */}
      {network.map((n, i) => (
        <Marker key={i} position={[n.lat, n.lng]} icon={lightIcon(n.signal)}>
          <Popup>
            <strong>{n.id}</strong>
            <br />
            Vehicles: {n.vehicles}
            <br />
            Queue: {n.queue_m} m<br />
            Signal: {n.signal}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
