/*
    run-at-time.js
    Class-based refactor for the run-at-time demo.
*/



function onRangeChange(r, f) {
    /* Source - https://stackoverflow.com/a/37623959
     * Posted by Andrew Willems, modified by community. See post 'Timeline' for change history
     * Retrieved 2025-11-11, License - CC BY-SA 4.0
    */
    var n, c, m;
    r.addEventListener("input", function (e) { n = 1; c = e.target.value; if (c != m) f(e); m = c; });
    r.addEventListener("change", function (e) { if (!n) f(e); });
}


class SimulationParams {
    constructor(formData) {
        this.t0 = parseFloat(formData.get('t0'));
        this.t1 = parseFloat(formData.get('t1'));
        this.n = parseInt(formData.get('N'));
        this.modelParams = {};

        for (let [key, value] of formData.entries()) {
            if (key !== 't0' && key !== 't1' && key !== 'N') {
                this.modelParams[key] = parseFloat(value);
            }
        }
    }

    isValid() {
        return !isNaN(this.t0) && !isNaN(this.t1) && !isNaN(this.n) && this.t1 > this.t0 && this.n > 0;
    }

    toPayload() {
        return {
            t0: this.t0,
            t1: this.t1,
            n: this.n,
            config: this.modelParams
        };
    }
}

class RunAtTimeDemo {
    constructor() {
        this.socket = null;
        this.chart = null;
        this.dom = {
            // DOM elements
            ranges: document.querySelectorAll('input[type=range]'),
            runForm: document.getElementById('runForm'),
            submitButtons: document.querySelectorAll("input[type=submit]"),
            messagesDiv: document.getElementById('messages'),
            canvas: document.getElementById('scatterPlot'),
        };
        this.config = {
            // wsUrl is now a global variable defined in the HTML template
            wsUrl: typeof wsUrl !== 'undefined' ? wsUrl : 'ws://localhost:8000',
        };

        this.initialize();
    }

    initialize() {
        this.setupChart();
        this.connectSocketIO();
        this.setupEventListeners();
        this.appendMessage('Demo initialized. Ready to run simulation.');
    }

    setupChart() {
        if (this.chart) {
            this.chart.destroy();
        }
        const ctx = this.dom.canvas.getContext('2d');
        const chartData = {
            datasets: [{
                label: 'Glucose Response Curve',
                data: [],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderWidth: 2,
                pointRadius: 1,
                fill: false,
                tension: 0.1, // Makes the line slightly curved
            }]
        };

        this.chart = new Chart(ctx, {
            type: 'line', // Changed from scatter to line
            data: chartData,
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: 'Glucose Response Curve'
                    }
                },
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Time (hours)' },
                        beginAtZero: true,
                    },
                    y: {
                        title: { display: true, text: 'Glucose Response Curve' },
                        beginAtZero: true
                    }
                },
                animation: false, // We handle updates manually for a progressive effect
            }
        });
    }

    connectSocketIO() {
        if (this.socket && this.socket.connected) {
            this.socket.disconnect();
        }

        this.socket = io(this.config.wsUrl, { transports: ['websocket'] });

        this.socket.on('connect', () => {
            this.appendMessage('Connected to Socket.IO server.');
        });

        this.socket.on('disconnect', () => {
            this.appendMessage('Socket.IO connection closed.');
        });

        this.socket.on('connect_error', (err) => {
            this.appendMessage('Socket.IO connection error: ' + err.message);
        });

        this.socket.on('message', (data) => {
            this.handleSocketMessage(data);
        });
    }

    handleSocketMessage(data) {
        try {
            const point = JSON.parse(data);
            if (point.error) {
                this.appendMessage(`Server Error: ${point.error}`);
                return;
            }
            if (typeof point.x !== 'undefined' && typeof point.y !== 'undefined') {
                this.chart.data.datasets[0].data.push(point);
                this.chart.update(); // Update the chart to draw the new point
            }
        } catch (e) {
            // This might just be a connection message, not an error.
            console.warn('Received non-JSON message:', data);
            this.appendMessage(`Received: ${data}`);
        }
    }

    setupEventListeners() {
        this.dom.runForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.runSimulation();
        });
        this.dom.submitButtons.forEach((button) => {
            button.addEventListener('click', () => {
                this.runSimulation();
            });
        });
        let self = this;
        // Range input listeners
        this.dom.ranges.forEach(range => {
            $(range).on("input change", function (e) {
                console.log(`Range ${range.id} updated to ${range.value}`);
                self.onUpdateRange(this);
            });
            $(range).on("click", function (e) { self.onUpdateRange(this); });
        });
    }

    get formData() {
        return new FormData(this.dom.runForm);
    }

    get simParams() {
        return new SimulationParams(this.formData);
    }

    async runSimulation() {
        if (!this.socket || !this.socket.connected) {
            this.appendMessage('Socket.IO not connected. Cannot send.');
            return;
        }

        // Clear previous results
        this.chart.data.datasets[0].data = [];
        this.chart.update();
        this.dom.messagesDiv.innerHTML = ''; // Clear messages
        this.appendMessage('Starting new simulation...');

        // Collect form data-derived simulation parameters
        const simParams = this.simParams;

        // Basic validation
        if (!simParams.isValid()) {
            this.appendMessage('Invalid simulation parameters. Please check t0, t1, and N.');
            return;
        }

        // Send run request
        const payload = simParams.toPayload();
        this.socket.emit('run', payload);
        this.appendMessage('Sent run request: ' + JSON.stringify(payload));
    }

    appendMessage(msg) {
        const el = document.createElement('div');
        el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        this.dom.messagesDiv.appendChild(el);
        this.dom.messagesDiv.scrollTop = this.dom.messagesDiv.scrollHeight;
    }

    onUpdateRange(range) {
        console.log(`Range ${range.id} updated to ${range.value}`);
        const outputElement = document.getElementById(`rangeValue-${range.id}`);
        if (range) {
            if (outputElement) {
                // Update the corresponding output element
                outputElement.textContent = range.value;
            } else {
                console.warn(`Element with id 'rangeValue-${range.id}' not found.`);
            }
            // Update the chart in real-time using the new range value
            const paramName = range.id;
            const paramValue = parseFloat(range.value);
            if (this.chart) {
                // For demonstration, let's say changing a range updates the chart title
                console.log(`Updating chart title with ${paramName}: ${paramValue}`);
                this.chart.options.plugins.title = {
                    display: true,
                    text: `Glucose Response Curve - ${paramName}: ${paramValue}`
                };
                this.chart.update();
            }
        }
    }
}

// Initialize the application once the DOM is fully loaded
var demoApp;
document.addEventListener('DOMContentLoaded', async () => {
    demoApp = new RunAtTimeDemo();
});
