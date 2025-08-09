import React, { useEffect, useState } from "react";
import { createRunAtTimeWebSocket } from "./websocket";
import { parseSampleData } from "./utils";
import { RunAtTimeChart } from "./RunAtTimeChart";

function App() {
  const [chartData, setChartData] = useState<any>({});
  const [rawMsg, setRawMsg] = useState<string[]>([]);

  useEffect(() => {
    const ws = createRunAtTimeWebSocket((msg) => {
      if (typeof msg === "object") {
        setChartData(parseSampleData(msg));
        setRawMsg((r) => [...r, "[Parsed sample data received]"]);
      } else {
        setRawMsg((r) => [...r, msg]);
      }
    });
    return () => ws.close();
  }, []);

  return (
    <div style={{ padding: "2em" }}>
      <h2>Run-at-Time WebSocket Chart Viewer</h2>
      <div>
        <RunAtTimeChart chartData={chartData} />
      </div>
      <div>
        <h3>Run</h3>
        <p>Connect to the WebSocket server to receive real-time data.</p>
        <p>Ensure the server is running at <code>ws://localhost:8001/ws/run-at-time</code>.</p>
        <button onClick={() => window.location.reload()}>Connect</button>
        <button onClick={() => setRawMsg([])}>Clear Messages</button>
        <button onClick={() => setChartData({})}>Clear Chart</button>
      </div>
      <div>
        <h3>Messages</h3>
        <ul>{rawMsg.map((msg, i) => <li key={i}>{msg}</li>)}</ul>
      </div>
    </div>
  );
}

export default App;
