"use client";

import { useEffect, useState } from "react";
import IntersectionNode from "./IntersectionNode";
import VehicleFlow from "./VehicleFlow";
import { initialNetwork } from "../data/mockNetwork";

const SIGNALS = ["RED", "YELLOW", "GREEN"];

export default function IntersectionNetwork() {
  const [network, setNetwork] = useState(initialNetwork);

  useEffect(() => {
    const timer = setInterval(() => {
      setNetwork((prev) =>
        prev.map((n) => ({
          ...n,
          signal: SIGNALS[Math.floor(Math.random() * SIGNALS.length)],
          vehicles: Math.max(0, n.vehicles + Math.floor(Math.random() * 4 - 1)),
        }))
      );
    }, 2000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="network">
      {/* ROW 1 */}
      <div className="row">
        <IntersectionNode {...network[0]} />
        <VehicleFlow signal={network[0].signal} />
        <IntersectionNode {...network[1]} />
        <VehicleFlow signal={network[1].signal} />
        <IntersectionNode {...network[2]} />
      </div>

      {/* ROW 2 */}
      <div className="row">
        <IntersectionNode {...network[3]} />
        <VehicleFlow signal={network[3].signal} />
        <IntersectionNode {...network[4]} />
        <VehicleFlow signal={network[4].signal} />
        <IntersectionNode {...network[5]} />
      </div>

      {/* ROW 3 */}
      <div className="row">
        <IntersectionNode {...network[6]} />
        <VehicleFlow signal={network[6].signal} />
        <IntersectionNode {...network[7]} />
      </div>
    </div>
  );
}
