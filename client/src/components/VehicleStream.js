"use client";

export default function VehicleStream({ path, density, signal, delay = 0 }) {
  if (signal !== "GREEN") return null;

  return [...Array(density)].map((_, i) => (
    <circle
      key={i}
      r="5"
      className="vehicle"
      style={{ animationDelay: `${i * delay}s` }}
    >
      <animateMotion
        dur={`${3 + density}s`}
        repeatCount="indefinite"
        path={path}
      />
    </circle>
  ));
}
