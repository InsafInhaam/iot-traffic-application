"use client";
import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import IntersectionMap from "../../components/IntersectionMap";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Dashboard() {
  const [network, setNetwork] = useState([]);
  const [chartData, setChartData] = useState([]);

  // --- Fetch network state ---
  useEffect(() => {
    const t = setInterval(async () => {
      const r = await fetch(API_BASE + "/api/v1/network");
      const d = await r.json();
      setNetwork(d.intersections || []);
    }, 2000);
    return () => clearInterval(t);
  }, []);

  // --- Fetch chart metrics ---
  useEffect(() => {
    const t = setInterval(async () => {
      const r = await fetch(API_BASE + "/api/v1/metrics/live");
      const d = await r.json();
      setChartData(d.points || []);
    }, 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="dashboard">
      <h1 className="title">🚦 Enterprise Traffic Analytics</h1>

      {/* ================= MAP ================= */}
      {/* <div className="map-box">
        <h2>🗺️ Intersection Network</h2>

        <div className="map">
          {network.map((n, i) => (
            <div key={i} className={`node node-${i}`}>
              <div className="tooltip">
                <strong>{n.id}</strong>
                <br />
                Vehicles: {n.vehicles}
                <br />
                Queue: {n.queue_m} m<br />
                Signal: {n.signal}
              </div>

              <div className="traffic-light">
                <span className={n.signal === "RED" ? "red on" : "red"} />
                <span
                  className={n.signal === "YELLOW" ? "yellow on" : "yellow"}
                />
                <span className={n.signal === "GREEN" ? "green on" : "green"} />
              </div>

              <div className="node-name">{n.id}</div>
            </div>
          ))}

          <div className="road h r1" />
          <div className="road h r2" />
          <div className="road v r3" />
          <div className="road v r4" />
        </div>
      </div> */}

      <IntersectionMap network={network} />

      {/* ================= CHART ================= */}
      <div className="chart-box">
        <h2>📈 Vehicles vs Time</h2>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line dataKey="vehicles" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
