"use client";

export default function TrafficLight({ signal }) {
  return (
    <div className="traffic-light">
      <span className={`red ${signal === "RED" ? "on" : ""}`} />
      <span className={`yellow ${signal === "YELLOW" ? "on" : ""}`} />
      <span className={`green ${signal === "GREEN" ? "on" : ""}`} />
    </div>
  );
}
