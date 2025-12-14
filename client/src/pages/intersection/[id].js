import { useRouter } from "next/router";
import useSWR from "swr";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const fetcher = (url) => fetch(url).then(r=>r.json());

export default function IntersectionView() {
  const router = useRouter();
  const { id } = router.query;
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const { data } = useSWR(id ? `${apiBase}/api/v1/intersection/${id}/latest` : null, fetcher, { refreshInterval: 3000 });

  // transform for chart: pick lane with most data
  if (!data) return <div>Loading...</div>;
  const laneNames = Object.keys(data);
  const lane = laneNames[0];
  const points = (data[lane] || []).map(d => ({ ts: new Date(d.ts).toLocaleTimeString(), queue_m: d.queue_m }));

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold">Intersection {id}</h1>
      <div className="mt-4">
        <h3 className="font-semibold">Lane {lane} recent queue (m)</h3>
        <div style={{ width: '100%', height: 300 }}>
          <ResponsiveContainer>
            <LineChart data={points}>
              <XAxis dataKey="ts" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="queue_m" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
