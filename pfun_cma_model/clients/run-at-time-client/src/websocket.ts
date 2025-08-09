import ReconnectingWebSocket from 'reconnecting-websocket';

export function createRunAtTimeWebSocket(onMessage: (data: any) => void) {
  const ws = new ReconnectingWebSocket('ws://localhost:8001/ws/run-at-time');

  ws.onopen = () => {
    console.log('WebSocket connected');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      // Handle plain text or error messages
      onMessage(event.data);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket closed');
  };

  ws.onerror = (err) => {
    console.error('WebSocket error', err);
  };

  return ws;
}
