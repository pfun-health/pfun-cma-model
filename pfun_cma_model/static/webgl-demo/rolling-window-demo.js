/*
    webgl-demo.js
    Adapted from run-at-time.js for the WebGL plot demo.
*/

/*
  From WebglPlotBundle:
  - WebglPlot
  - WebglLine
  - ColorRGBA
*/

class RollingWindowDemo {
    constructor() {
        this.socket = null;
        this.wglp = null;
        this.line = null;
        this.dom = {
            ranges: document.querySelectorAll('input[type=range]'),
            runForm: document.getElementById('runForm'),
            submitButtons: document.querySelectorAll("input[type=submit]"),
            messagesDiv: document.getElementById('messages'),
            canvas: document.getElementById('webgl-plot'),
        };
        this.config = {
            wsUrl: typeof wsUrl !== 'undefined' ? wsUrl : 'ws://localhost:8000',
        };

        this.initialize();
    }

    initialize() {
        this.connectSocketIO();
        this.setupEventListeners();
        this.appendMessage('Demo initialized. Ready to run simulation.');
        this.setupPlot(500); // Initial plot setup with a window size of 500
    }

    setupPlot(windowSize) {
        if (this.wglp) {
            this.wglp.removeAllLines();
        }

        const devicePixelRatio = window.devicePixelRatio || 1;
        this.dom.canvas.width = this.dom.canvas.clientWidth * devicePixelRatio;
        this.dom.canvas.height = this.dom.canvas.clientHeight * devicePixelRatio;

        const color = new WebglPlotBundle.ColorRGBA(0.21, 0.64, 0.88, 1);
        this.line = new WebglPlotBundle.WebglLine(color, windowSize);
        this.line.arrangeX();

        // Initialize with a horizontal line at y = 0
        for (let i = 0; i < windowSize; i++) {
            this.line.setY(i, 0);
        }

        if (!this.wglp) {
            this.wglp = new WebglPlotBundle.WebglPlot(this.dom.canvas);
        }

        this.wglp.addLine(this.line);

        // Animation loop
        const newFrame = () => {
            if (this.wglp) {
                this.wglp.update();
            }
            requestAnimationFrame(newFrame);
        }
        requestAnimationFrame(newFrame);
    }

    connectSocketIO() {
        if (this.socket && this.socket.connected) {
            this.socket.disconnect();
        }

        this.socket = io(this.config.wsUrl, { transports: ['websocket'] });

        this.socket.on('connect', () => this.appendMessage('Connected to Socket.IO server.'));
        this.socket.on('disconnect', () => this.appendMessage('Socket.IO connection closed.'));
        this.socket.on('connect_error', (err) => this.appendMessage(`Socket.IO connection error: ${err.message}`));
        this.socket.on('message', (data) => this.handleSocketMessage(data));
    }

    handleSocketMessage(data) {
        try {
            const point = JSON.parse(data);
            if (point.error) {
                this.appendMessage(`Server Error: ${point.error}`);
                return;
            }
            if (typeof point.y !== 'undefined' && this.line) {
                const scaledY = (point.y - 0.5); // scale glucose value as needed
                this.line.shiftAdd(new Float32Array([scaledY]));
            }
        } catch (e) {
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
            button.addEventListener('click', (e) => {
                e.preventDefault();
                this.runSimulation();
            });
        });

        this.dom.ranges.forEach(range => {
            range.addEventListener('input', () => this.onUpdateRange(range));
        });
    }

    runSimulation() {
        if (!this.socket || !this.socket.connected) {
            this.appendMessage('Socket.IO not connected. Cannot send.');
            return;
        }

        this.dom.messagesDiv.innerHTML = '';
        this.appendMessage('Starting new simulation...');

        const formData = new FormData(this.dom.runForm);
        const t0 = parseFloat(formData.get('t0'));
        const t1 = parseFloat(formData.get('t1'));
        const n = parseInt(formData.get('N'));

        if (isNaN(t0) || isNaN(t1) || isNaN(n) || t1 <= t0 || n <= 0) {
            this.appendMessage('Invalid simulation parameters. Please check t0, t1, and N.');
            return;
        }

        // Plot setup is now fixed, no need to re-setup for simulation

        const modelParams = {};
        for (let [key, value] of formData.entries()) {
            if (key !== 't0' && key !== 't1' && key !== 'N') {
                modelParams[key] = parseFloat(value);
            }
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
        if (range && outputElement) {
            outputElement.textContent = range.value;
        }
    }
}
