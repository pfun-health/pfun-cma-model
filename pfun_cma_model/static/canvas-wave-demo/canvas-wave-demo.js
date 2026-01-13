/*
    canvas-wave-demo.js
    Interactive canvas demo for visualizing CMA model output.
*/

class MousePosVector {
    constructor(x = 0.5, y = 0.5) {
        // default is center of canvas
        this.x = x;
        this.xp = null;
        this.y = y;
        this.yp = null;
    }

    update(x, y) {
        
        this.xp = this.x;
        this.yp = this.y;
        this.x = x;
        this.y = y;
    }
};

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
            BInput: document.getElementById('B'),
            taugInput: document.getElementById('taug'),
        };
        this.c = this.dom.canvas.getContext("2d");
        this.config = {
            wsUrl: typeof wsUrl !== 'undefined' ? wsUrl : 'ws://localhost:8000',
        };
        this.mouseDown = false;
        this.mousePos = new MousePosVector(0.5, 0.5);
        this.initialize();
    }

    initialize() {
        this.connectSocketIO();
        this.setupEventListeners();
        this.appendMessage('Demo initialized. Drag on the canvas to start.');
        this.resizeCanvas();
        this.draw();
        try {
            this.runSimulation();
        } catch (e) {
            this.appendMessage('Error starting initial simulation: ' + e.message);
        }
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
        if (!this.dom.BInput || !this.dom.taugInput) {
            this.appendMessage('Error: B or taug inputs not found.');
            return;
        }
        const rect = this.dom.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        // console.debug("Left:", rect.left, "clientX:", e.clientX, "Top:", rect.top, "clientY:", e.clientY);

        const BInputRange = parseFloat(this.dom.BInput.max) - parseFloat(this.dom.BInput.min);
        const taugInputRange = parseFloat(this.dom.taugInput.max) - parseFloat(this.dom.taugInput.min);

        const compute_x_delta = (x, Xmin, Xrange, canvasWidth) => {
            return parseFloat(Xmin) + (x / canvasWidth) * Xrange;
        }

        const compute_y_delta = (y, Ymin, Yrange, canvasHeight) => {
            return parseFloat(Ymin) + (1 - y / canvasHeight) * Yrange;
        }

        const newBInput = compute_x_delta(x, this.dom.BInput.min, BInputRange, this.dom.canvas.width);
        const newtaugInput = compute_y_delta(y, this.dom.taugInput.min, taugInputRange, this.dom.canvas.height);

        // validation for NaN
        if (isNaN(newBInput) || isNaN(newtaugInput)) {
            this.appendMessage('Error: Computed B or taug values are invalid.');
            return;
        }
        // validation for out-of-bounds
        if (newBInput < parseFloat(this.dom.BInput.min) || newBInput > parseFloat(this.dom.BInput.max) ||
            newtaugInput < parseFloat(this.dom.taugInput.min) || newtaugInput > parseFloat(this.dom.taugInput.max)) {
            this.appendMessage('Error: Computed B or taug values are out of range.');
            return;
        }
        // validation for significant change
        if (Math.abs(newBInput - parseFloat(this.dom.BInput.value)) < 0.025 &&
            Math.abs(newtaugInput - parseFloat(this.dom.taugInput.value)) < 0.025) {
            // No significant change
            return;
        }
        // Update input values
        this.dom.BInput.value = newBInput.toFixed(2);
        this.dom.taugInput.value = newtaugInput.toFixed(2);

        // Update slider output values
        document.getElementById('rangeValue-B').textContent = newBInput.toFixed(2);
        document.getElementById('rangeValue-taug').textContent = newtaugInput.toFixed(2);

        // run the simulation (from the current mouse event)
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

        /*
            Begin drawing the path to the canvas.
        */
        this.c.beginPath();

        // Set path style attributes (color, linewidth, ...)
        this.c.strokeStyle = 'cyan';
        this.c.fillStyle = 'rgba(255, 0, 0, 0.82)';
        this.c.lineWidth = 2;

        let minDrawX = Infinity;
        let maxDrawX = -Infinity;
        let minDrawY = Infinity;
        let maxDrawY = -Infinity;
        for (let i = 0; i < this.cells.length; i++) {
            const p = this.cells[i];
            const x = (p.x - minX) / (maxX - minX) * this.dom.canvas.width;
            const y = this.dom.canvas.height - (p.y - minY) / (maxY - minY) * this.dom.canvas.height;
            minDrawX = Math.min(minDrawX, x);
            maxDrawX = Math.max(maxDrawX, x);
            minDrawY = Math.min(minDrawY, y);
            maxDrawY = Math.max(maxDrawY, y);
            if (i === 0) {
                this.c.moveTo(x, y);
            } else {
                this.c.lineTo(x, y);
            }
        }
        // before closing, add a final point in the lower right corner
        this.c.lineTo(maxDrawX, maxDrawY);
        // finally, close the path, draw the stroke, and fill
        this.c.closePath();
        this.c.stroke();
        this.c.fill();
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

var demoApp;
document.addEventListener('DOMContentLoaded', () => {
    demoApp = new CanvasWaveDemo();
});
