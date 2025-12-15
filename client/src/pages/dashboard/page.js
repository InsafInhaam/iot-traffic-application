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
import LiveCard from "../../components/LiveCard";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Dashboard() {
  const [network, setNetwork] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [events, setEvents] = useState([]);
  const [chartData, setChartData] = useState([]);

  /* ---------- DATA POLLING ---------- */
  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const e = await fetch(`${API_BASE}/api/v1/events/latest`).then((r) =>
          r.json()
        );

        setEvents(e.events || []);
      } catch (err) {
        console.error("API error", err);
      }
    }, 2000);

    return () => clearInterval(timer);
  }, []);

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

  /* ---------- KPIs ---------- */
  const totalIntersections = network.length;

  const totalVehicles = network.reduce((sum, i) => sum + (i.vehicles || 0), 0);

  const avgCongestion =
    totalIntersections === 0
      ? 0
      : Math.round(
          network.reduce((s, i) => s + (i.queue_m || 0), 0) / totalIntersections
        );

  return (
    <div className="dashboard">
      <h1 className="title">🚦 Enterprise Traffic Analytics</h1>

      {/* ================= KPI CARDS ================= */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPI title="Intersections" value={totalIntersections} />
        <KPI title="Total Vehicles" value={totalVehicles} />
        <KPI title="Avg Congestion (m)" value={avgCongestion} />
      </div>

      {/* ================= MAP ================= */}
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

      {/* ================= LIVE EVENTS ================= */}
      <div className="bg-white rounded-xl shadow p-4">
        <h2 className="font-semibold mb-3">🚨 Live Events</h2>

        <ul className="space-y-2 max-h-64 overflow-auto text-sm">
          {events.length === 0 && (
            <li className="text-gray-500">No events yet</li>
          )}

          {events.map((e, i) => (
            <li key={i} className="flex justify-between border-b pb-1">
              <span>{e.intersection}</span>
              <span className="text-gray-500">{e.time}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/* ================= KPI CARD ================= */
function KPI({ title, value }) {
  return (
    <div className="bg-white rounded-xl shadow p-5">
      <p className="text-gray-500">{title}</p>
      <p className="text-3xl font-bold">{value}</p>
    </div>
  );
}
