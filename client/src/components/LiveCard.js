import { useEffect, useState } from "react";

export default function LiveCard({ wsUrl }) {
  const [events, setEvents] = useState([]);
  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => console.log("ws open");
    ws.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        setEvents((s) => [d, ...s].slice(0, 20));
      } catch (e) {}
    };
    ws.onclose = () => console.log("ws closed");
    return () => ws.close();
  }, [wsUrl]);

  return (
    <div className="p-4 border rounded">
      <h3 className="font-bold">Live Events</h3>
      <ul>
        {events.map((e, i) => (
          <li key={i} className="text-sm">
            <b>{e.type || e.payload?.type}</b> —{" "}
            {e.intersection || e.payload?.intersection || ""}{" "}
            <span className="text-xs text-gray-500">
              {new Date().toLocaleTimeString()}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
