/*
    run-at-time.js
    Class-based refactor for the run-at-time demo.
*/

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
        this.dom.ranges.forEach(range => {
            range.addEventListener('input', () => {
                self.onUpdateRange(range);
            });
        });
    }

    runSimulation() {
        if (!this.socket || !this.socket.connected) {
            this.appendMessage('Socket.IO not connected. Cannot send.');
            return;
        }

        // Clear previous results
        this.chart.data.datasets[0].data = [];
        this.chart.update();
        this.dom.messagesDiv.innerHTML = ''; // Clear messages
        this.appendMessage('Starting new simulation...');

        // Collect form data
        const formData = new FormData(this.dom.runForm);
        const t0 = parseFloat(formData.get('t0'));
        const t1 = parseFloat(formData.get('t1'));
        const n = parseInt(formData.get('N'));

        const modelParams = {};
        for (let [key, value] of formData.entries()) {
            if (key !== 't0' && key !== 't1' && key !== 'N') {
                modelParams[key] = parseFloat(value);
            }
        }

        // Basic validation
        if (isNaN(t0) || isNaN(t1) || isNaN(n) || t1 <= t0 || n <= 0) {
            this.appendMessage('Invalid simulation parameters. Please check t0, t1, and N.');
            return;
        }

        const payload = { t0, t1, n, config: modelParams };
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
        const outputElement = document.getElementById(`rangeValue-${range.id}`);
        if (range) {
            if (outputElement) {
                outputElement.textContent = range.value;
            } else {
                console.warn(`Element with id 'rangeValue-${range.id}' not found.`);
            }
        }
    }
}

// Initialize the application once the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    new RunAtTimeDemo();
});