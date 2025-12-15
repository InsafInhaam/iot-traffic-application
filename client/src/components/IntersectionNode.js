"use client";

import TrafficLight from "./TrafficLight";

export default function IntersectionNode({ id, signal, vehicles }) {
  return (
    <div className="node">
      <TrafficLight signal={signal} />

      <div className="node-info">
        <strong>Intersection {id}</strong>
        <div>🚗 Vehicles: {vehicles}</div>
      </div>
    </div>
  );
}
