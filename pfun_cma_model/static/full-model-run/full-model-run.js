/*
    full-model-run.js
    Class-based implementation for the full-model-run demo.
*/

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

class FullModelRunDemo {
    constructor() {
        this.socket = null;
        this.chart = null;
        this.dom = {
            // DOM elements
            ranges: document.querySelectorAll('input[type=range]'),
            runForm: document.getElementById('runForm'),
            submitButtons: document.querySelectorAll("input[type=submit]"),
            messagesDiv: document.getElementById('messages'),
            canvas: document.getElementById('chartCanvas'),
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
        try {
            demoApp.runSimulation();
        } catch (err) {
            console.error('Failed to start initial simulation.', err);
        }
    }

    setupChart() {
        if (this.chart) {
            this.chart.destroy();
        }
        const ctx = this.dom.canvas.getContext('2d');
        const chartData = {
            datasets: [
                {
                    label: 'Cortisol (c)',
                    data: [],
                    borderColor: 'cyan',
                    backgroundColor: 'cyan',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: 'Melatonin (m)',
                    data: [],
                    borderColor: 'darkgrey',
                    backgroundColor: 'darkgrey',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: 'Adiponectin (a)',
                    data: [],
                    borderColor: 'magenta', // or #ec5ef9 based on python code
                    backgroundColor: 'magenta',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1,
                }
            ]
        };

        this.chart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: 'CMA Model Output'
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
                        title: { display: true, text: 'Relative Amplitude' },
                        beginAtZero: true
                    }
                },
                animation: false,
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
            // Point is expected to be {t: ..., c: ..., m: ..., a: ...}
            if (typeof point.t !== 'undefined') {
                this.chart.data.datasets[0].data.push({x: point.t, y: point.c});
                this.chart.data.datasets[1].data.push({x: point.t, y: point.m});
                this.chart.data.datasets[2].data.push({x: point.t, y: point.a});
            }
            // Update every point or throttling if needed, for now every point as it's not too fast
            // or maybe every 10 points like the other demo
            this.chart.update();
        } catch (e) {
            console.warn('Received non-JSON message or error parsing:', data);
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
            // Update display value on load
            self.onUpdateRange(range, false);

            range.addEventListener("input", function (e) {
                self.onUpdateRange(this);
            });
            // Also trigger on change just in case
            range.addEventListener("change", function (e) {
                 // self.onUpdateRange(this); // duplicative if input handles it
            });
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
        (async () => {
            this.chart.data.datasets.forEach(ds => ds.data = []);
            setTimeout(() => {
                this.chart.update();
            }, 50);
        })();
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
        // Emit 'run_full' instead of 'run'
        this.socket.emit('run_full', payload);
        this.appendMessage('Sent run request: ' + JSON.stringify(payload));
    }

    appendMessage(msg) {
        const el = document.createElement('div');
        el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        this.dom.messagesDiv.appendChild(el);
        this.dom.messagesDiv.scrollTop = this.dom.messagesDiv.scrollHeight;
    }

    onUpdateRange(range, updateChart = true) {
        const outputElement = document.getElementById(`rangeValue-${range.id}`);
        if (range) {
            if (outputElement) {
                // Update the corresponding output element
                outputElement.textContent = range.value;
            }

            // Auto-run logic could go here if we want real-time updates while dragging
            // For now, let's keep it manual run or on change
            if (updateChart && this.chart) {
                 setTimeout(() => {
                    // Debounce or check logic if needed
                    // For now, just re-run if it's a "quick" update
                     if(parseFloat(`${(new Date()).getTime()}`.at(-1)) < 5) {
                        this.runSimulation();
                    }
                }, 100);
            }
        }
    }
}

// Initialize the application once the DOM is fully loaded
var demoApp;
document.addEventListener('DOMContentLoaded', async () => {
    demoApp = new FullModelRunDemo();
});
