"use client";

import TrafficLight from "./TrafficLight";
import VehicleStream from "./VehicleStream";
import { rushMultiplier, heatColor } from "../utils/trafficLogic";

export default function TrafficMap({ network }) {
  const rush = rushMultiplier();

  return (
    <svg viewBox="0 0 800 500" className="traffic-map">
      {/* HEATMAP ROADS */}
      <path
        d="M100 100 H700"
        stroke={heatColor(network[0].vehicles * rush)}
        className="heat-road"
      />
      <path
        d="M100 250 H700"
        stroke={heatColor(network[3].vehicles * rush)}
        className="heat-road"
      />
      <path
        d="M200 50 V450"
        stroke={heatColor(network[1].vehicles * rush)}
        className="heat-road"
      />
      <path
        d="M400 50 V450"
        stroke={heatColor(network[4].vehicles * rush)}
        className="heat-road"
      />

      {/* BASE ROADS */}
      <path d="M100 100 H700" className="road" />
      <path d="M100 250 H700" className="road" />
      <path d="M200 50 V450" className="road" />
      <path d="M400 50 V450" className="road" />

      {/* VEHICLE FLOWS (TURNING LANES) */}
      <VehicleStream
        path="M100 100 H700"
        density={Math.floor(network[0].vehicles * rush)}
        signal={network[0].signal}
        delay={0.4}
      />

      {/* LEFT TURN */}
      <VehicleStream
        path="M400 100 C400 180 320 220 200 250"
        density={Math.floor(network[1].vehicles * rush * 0.6)}
        signal={network[1].signal}
        delay={0.6}
      />

      {/* RIGHT TURN */}
      <VehicleStream
        path="M400 250 C480 300 550 300 600 250"
        density={Math.floor(network[4].vehicles * rush * 0.6)}
        signal={network[4].signal}
        delay={0.6}
      />

      {/* INTERSECTIONS + LIGHTS */}
      {[
        [200, 100, network[0]],
        [400, 100, network[1]],
        [600, 100, network[2]],
        [200, 250, network[3]],
        [400, 250, network[4]],
        [600, 250, network[5]],
      ].map(([x, y, n], i) => (
        <g key={i} transform={`translate(${x},${y})`}>
          <circle r="18" className="intersection" />
          <TrafficLight signal={n.signal} />
        </g>
      ))}
    </svg>
  );
}
