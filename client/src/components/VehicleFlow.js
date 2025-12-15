"use client";

export default function VehicleFlow({ direction = "horizontal", signal }) {
  return (
    <div className={`vehicle-lane ${direction}`}>
      <div className={`vehicle ${signal === "GREEN" ? "move" : "stop"}`} />
    </div>
  );
}
