/*
    run-at-time.js
*/


const messagesDiv = document.getElementById('messages');
let socket;
// Parse params from template variable
const params = new URLSearchParams(window.location.href);

// Chart.js scatter plot setup
const canvas = document.getElementById('scatterPlot');
// Set explicit pixel size to avoid Chart.js infinite resize bug
canvas.width = window.visualViewport.width * window.devicePixelRatio;
canvas.height = window.visualViewport.height * window.devicePixelRatio;
const ctx = canvas.getContext('2d');
const scatterData = {
    datasets: [{
        label: 'Glucose Response Curve',
        data: [],
        backgroundColor: 'rgba(54, 162, 235, 0.7)',
        pointRadius: 2,
    }]
};
// Custom plugin for onioning/fade effect
const onionFadePlugin = {
    id: 'onionFade',
    beforeDatasetsDraw(chart, args, options) {
        const ctx = chart.ctx;
        chart.data.datasets.forEach((dataset, i) => {
            dataset.data.forEach((pt, idx) => {
                if (pt._fade) {
                    const meta = chart.getDatasetMeta(i);
                    const point = meta.data[idx];
                    if (point) {
                        ctx.save();
                        ctx.globalAlpha = 0.5;
                        point.draw(ctx);
                        ctx.restore();
                    }
                }
            });
        });
    }
};

const fadeDuration = 250; // ms
const scatterChart = new Chart(ctx, {
    type: 'scatter',
    data: scatterData,
    options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: window.visualViewport.width / window.visualViewport.height,
        plugins: {
            onionFade: {
                fadeDuration: fadeDuration // Duration for fade effect in ms
            },
        },
        scales: {
            x: {
                type: 'linear',
                title: { display: true, text: 'Time (hours)' },
                beginAtZero: true,
            },
            y: {
                title: { display: true, text: 'Glucose Response Curve (normalised [0.0, 2.0])' },
                beginAtZero: true
            }
        }
    },
    plugins: [onionFadePlugin]
});

function appendMessage(msg) {
    const el = document.createElement('div');
    el.textContent = msg;
    messagesDiv.appendChild(el);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function addScatterPoints(points) {
    // Smoothly transition to new points using onioning effect
    // Keep previous points and fade them out, fade in new points
    const prevData = scatterData.datasets[0].data.slice();
    let newData;
    if (Array.isArray(points)) {
        newData = points;
    } else if (typeof points === 'object' && points !== null && 'x' in points && 'y' in points) {
        newData = [points];
    } else {
        console.error('Invalid points format:', points);
        return;
    }

    // Onioning: overlay previous points with alpha, fade in new points
    scatterData.datasets[0].data = prevData.map(pt => ({ ...pt, _fade: false }));
    scatterChart.update('active');
    await (async () => {
        await new Promise(resolve => setTimeout(() => {
            scatterData.datasets[0].data = newData.map(pt => ({ ...pt, _fade: true }));
            scatterChart.update('active');
        }, fadeDuration) // Wait for fade effect
        );
    })();
}

function createScatterOnePoint(xdatum, ydatum) {
    /* Create a single scatter plot point datum. */
    return { x: xdatum, y: ydatum };
}

function createScatterPointArrayFromDataObject(dataObject) {
    /*
    Create an array of scatter plot points from a data object formatted as {key: value, ...}.
    The keys will be used as the x-coordinates and the values as the y-coordinates.
    ({key: value}) -> {x: key, y: value}
    */
    if (typeof (dataObject) === 'object' && dataObject !== null) {
        return Object.entries(dataObject).map(([key, item]) => createScatterOnePoint(key, item));
    } else if (Array.isArray(dataObject)) {
        return dataObject.map((item, index) => createScatterOnePoint(index, item));
    }
}


// WebSocket connection setup

let wsUrl; // Declare wsUrl as a mutable variable

async function initializeWebSocket() {
    const response = await fetch('/ws/config');
    const wsConfig = await response.json();
    const wsProtocol = wsConfig.scheme;
    const wsHost = wsConfig.host;
    const wsPort = wsConfig.port;
    // Construct the WebSocket URL
    wsUrl = `${wsProtocol}://${wsHost}:${wsPort}`;
    connectSocketIO(); // Now call connectSocketIO without passing wsUrl, as it's global
}

function connectSocketIO() { // No longer takes wsUrl as an argument
    if (socket && socket.connected) {
        socket.disconnect();
    }
    // Socket.IO expects the base URL, not the full ws path
    // We'll use the namespace/endpoint as the path

    socket = io(wsUrl, { transports: ['websocket'] });

    socket.on('connect', () => {
        appendMessage('Connected to Socket.IO server.');
    });

    socket.on('disconnect', () => {
        appendMessage('Socket.IO connection closed.');
    });

    socket.on('connect_error', (err) => {
        appendMessage('Socket.IO connection error: ' + err.message);
    });

    // Listen for messages from server (event name may need to match backend)
    socket.on('message', (data) => {
        appendMessage('Received: ' + JSON.stringify(data));
        try {
            // If data is a string, try to parse as JSON
            let parsed = data;
            if (typeof data === 'string') {
                parsed = JSON.parse(data);
            }
            for (const key in parsed) {
                if (parsed.hasOwnProperty(key)) {
                    let scatterPoints = createScatterPointArrayFromDataObject(parsed[key]);
                    addScatterPoints(scatterPoints);
                }
            }
        } catch (e) {
            console.warn('Socket.IO message not valid JSON or not a point:', e);
        }
    });
}

// Helper: trigger chart transition and websocket reconnect
async function triggerChartUpdateAndReconnect() {
    // Onion fade out current points
    // Mark current points for fading
    scatterData.datasets[0].data = scatterData.datasets[0].data.map(pt => ({ ...pt, _fade: true }));
    scatterChart.update('active'); // Update to show faded points

    // Wait for a short duration to allow the fade effect to be visible
    await new Promise(resolve => setTimeout(resolve, fadeDuration * 0.5));

    // Clear the data after the fade out
    scatterData.datasets[0].data = [];
    scatterChart.update('active'); // Update to clear the chart

    // No explicit "reconnect" logic here, as the `initializeWebSocket`
    // and `connectSocketIO` functions handle the connection state.
    // The `chart_update_hook` will eventually send new data,
    // which will then be added to the chart.
}

// Initial connection
initializeWebSocket(); // Call the new initialization function

// Event-driven: update on parameter change
document.querySelectorAll('#runForm input .model-params').forEach(input => {
    input.addEventListener('change', () => {
        triggerChartUpdateAndReconnect();
    });
});

// activated when the relevant parameters change (or when the user clicks 'submit')
function chart_update_hook(e) {
    e.preventDefault();
    triggerChartUpdateAndReconnect();
    setTimeout(() => {
        if (!socket || !socket.connected) {
            appendMessage('Socket.IO not connected. Cannot send.');
            return;
        }
        const t0 = parseFloat(document.getElementById('t0').value);
        const t1 = parseFloat(document.getElementById('t1').value);
        const N = parseInt(document.getElementById('N').value);

        // Basic validation
        if (isNaN(t0) || t0 < 0 || t0 > 24) {
            appendMessage('Invalid t0');
            return;
        }
        if (isNaN(t1) || t1 < 0 || t1 > 24) {
            appendMessage('Invalid t1');
            return;
        }
        if (isNaN(N) || N < 1 || N > 10000) {
            appendMessage('Invalid N');
            return;
        }

        const config = {};
        params.forEach((_, key) => {
            const input = document.getElementById(key);
            if (input) {
                let value = input.value;
                if (!isNaN(value) && value.trim() !== '') {
                    value = parseFloat(value);
                }
                config[key] = value;
            }
        });

        const payload = { t0, t1, n: N, config };
        // Use a custom event name, e.g. 'run', or 'run-at-time', as expected by backend
        socket.emit('run', payload);
        appendMessage('Sent: ' + JSON.stringify(payload));
    }, fadeDuration); // Wait for fade out and reconnect
}

// setup chart_update_hook, triggered on 'submit' or 'input change'
document.getElementById('runForm').onsubmit = chart_update_hook;
document.querySelectorAll('#runForm input .model-params').forEach(input => {
    input.addEventListener('change', chart_update_hook);
});
