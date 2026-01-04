/*
    canvas-wave-demo.js
    Interactive canvas demo for visualizing CMA model output.
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

class CanvasWaveDemo {
    constructor() {
        this.socket = null;
        this.cells = [];
        this.dom = {
            runForm: document.getElementById('runForm'),
            messagesDiv: document.getElementById('messages'),
            canvas: document.getElementById('waveCanvas'),
            cPeakInput: document.getElementById('c_peak'),
            mPeakInput: document.getElementById('m_peak'),
        };
        this.c = this.dom.canvas.getContext("2d");
        this.config = {
            wsUrl: typeof wsUrl !== 'undefined' ? wsUrl : 'ws://localhost:8000',
        };
        this.mouseDown = false;

        this.initialize();
    }

    initialize() {
        this.connectSocketIO();
        this.setupEventListeners();
        this.appendMessage('Demo initialized. Drag on the canvas to start.');
        this.resizeCanvas();
        this.draw();
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
                this.cells.push(point);
            }
        } catch (e) {
            console.warn('Received non-JSON message:', data);
        }
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.resizeCanvas());

        this.dom.canvas.addEventListener("mousedown", (e) => {
            this.mouseDown = true;
            this.runSimulationFromMouseEvent(e);
        });

        this.dom.canvas.addEventListener("mousemove", (e) => {
            if (this.mouseDown) {
                this.runSimulationFromMouseEvent(e);
            }
        });

        this.dom.canvas.addEventListener("mouseup", (e) => {
            this.mouseDown = false;
        });

        this.dom.runForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.runSimulation();
        });
    }

    resizeCanvas() {
        this.dom.canvas.width = this.dom.canvas.offsetWidth;
        this.dom.canvas.height = this.dom.canvas.offsetHeight;
    }

    get formData() {
        return new FormData(this.dom.runForm);
    }

    get simParams() {
        return new SimulationParams(this.formData);
    }

    runSimulationFromMouseEvent(e) {
        const rect = this.dom.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const cPeakRange = parseFloat(this.dom.cPeakInput.max) - parseFloat(this.dom.cPeakInput.min);
        const mPeakRange = parseFloat(this.dom.mPeakInput.max) - parseFloat(this.dom.mPeakInput.min);

        const newCPeak = parseFloat(this.dom.cPeakInput.min) + (x / this.dom.canvas.width) * cPeakRange;
        const newMPeak = parseFloat(this.dom.mPeakInput.min) + (1 - y / this.dom.canvas.height) * mPeakRange;

        this.dom.cPeakInput.value = newCPeak.toFixed(2);
        this.dom.mPeakInput.value = newMPeak.toFixed(2);

        // Update slider output values
        document.getElementById('rangeValue-c_peak').textContent = newCPeak.toFixed(2);
        document.getElementById('rangeValue-m_peak').textContent = newMPeak.toFixed(2);

        this.runSimulation();
    }

    runSimulation() {
        if (!this.socket || !this.socket.connected) {
            this.appendMessage('Socket.IO not connected. Cannot send.');
            return;
        }

        this.cells = []; // Clear previous data
        this.appendMessage('Starting new simulation...');

        const simParams = this.simParams;
        if (!simParams.isValid()) {
            this.appendMessage('Invalid simulation parameters.');
            return;
        }

        const payload = simParams.toPayload();
        this.socket.emit('run', payload);
        this.appendMessage('Sent run request: ' + JSON.stringify(payload.config, null, 2));
    }

    draw() {
        requestAnimationFrame(() => this.draw());

        this.c.clearRect(0, 0, this.dom.canvas.width, this.dom.canvas.height);

        if (this.cells.length < 2) return;
        
        // Find min/max for scaling
        let minX = this.cells[0].x, maxX = this.cells[0].x;
        let minY = this.cells[0].y, maxY = this.cells[0].y;
        for(let i = 1; i < this.cells.length; i++) {
            minX = Math.min(minX, this.cells[i].x);
            maxX = Math.max(maxX, this.cells[i].x);
            minY = Math.min(minY, this.cells[i].y);
            maxY = Math.max(maxY, this.cells[i].y);
        }

        // Add some padding to y-axis
        const yPadding = (maxY - minY) * 0.1;
        minY -= yPadding;
        maxY += yPadding;
        if (maxY === minY) {
             maxY +=1;
             minY -=1;
        }

        this.c.beginPath();
        this.c.strokeStyle = 'cyan';
        this.c.lineWidth = 2;

        for (let i = 0; i < this.cells.length; i++) {
            const p = this.cells[i];
            const x = (p.x - minX) / (maxX - minX) * this.dom.canvas.width;
            const y = this.dom.canvas.height - (p.y - minY) / (maxY - minY) * this.dom.canvas.height;
            if (i === 0) {
                this.c.moveTo(x, y);
            } else {
                this.c.lineTo(x, y);
            }
        }
        this.c.stroke();
    }

    appendMessage(msg) {
        if(this.dom.messagesDiv.childElementCount > 100) {
            this.dom.messagesDiv.removeChild(this.dom.messagesDiv.firstChild);
        }
        const el = document.createElement('div');
        el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        this.dom.messagesDiv.appendChild(el);
        this.dom.messagesDiv.scrollTop = this.dom.messagesDiv.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new CanvasWaveDemo();
});
