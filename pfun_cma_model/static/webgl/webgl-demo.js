class WebGLDemo {
    constructor() {
        this.socket = null;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer();
        this.particles = [];
        this.particleSystem = null;

        this.dom = {
            container: document.getElementById('webgl-container'),
        };

        this.config = {
            wsUrl: typeof wsUrl !== 'undefined' ? wsUrl : 'ws://localhost:8000',
        };

        this.initialize();
    }

    initialize() {
        this.setupRenderer();
        this.setupScene();
        this.connectSocketIO();
        this.animate();
        window.addEventListener('resize', this.onWindowResize.bind(this), false);
    }

    setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.dom.container.appendChild(this.renderer.domElement);
    }

    setupScene() {
        this.camera.position.z = 5;
    }

    connectSocketIO() {
        this.socket = io(this.config.wsUrl, { transports: ['websocket'] });

        this.socket.on('connect', () => {
            console.log('Connected to Socket.IO server.');
            this.runSimulation(); // Automatically start the simulation
        });

        this.socket.on('disconnect', () => {
            console.log('Socket.IO connection closed.');
        });

        this.socket.on('connect_error', (err) => {
            console.error('Socket.IO connection error:', err.message);
        });

        this.socket.on('message', (data) => {
            this.handleSocketMessage(data);
        });
    }

    handleSocketMessage(data) {
        try {
            const point = JSON.parse(data);
            if (point.error) {
                console.error('Server Error:', point.error);
                return;
            }
            if (typeof point.x !== 'undefined' && typeof point.y !== 'undefined') {
                this.addParticle(point);
            }
        } catch (e) {
            console.warn('Received non-JSON message:', data);
        }
    }

    addParticle(point) {
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array([0, 0, 0]);
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

        const material = new THREE.PointsMaterial({
            color: new THREE.Color(Math.random(), Math.random(), Math.random()),
            size: 0.1,
            transparent: true,
            opacity: 1.0,
        });

        const particle = new THREE.Points(geometry, material);
        particle.position.x = (point.x - 12) / 2; // Normalize x to be around 0
        particle.position.y = (point.y - 100) / 20; // Normalize y
        particle.position.z = Math.random() * 2 - 1;

        particle.velocity = new THREE.Vector3(
            (Math.random() - 0.5) * 0.01,
            (Math.random() - 0.5) * 0.01,
            (Math.random() - 0.5) * 0.01
        );

        particle.createdAt = Date.now();

        this.scene.add(particle);
        this.particles.push(particle);
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));

        const now = Date.now();
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];

            // Update position
            particle.position.add(particle.velocity);

            // Fade out
            const age = now - particle.createdAt;
            if (age > 2000) {
                particle.material.opacity -= 0.05;
            }

            // Remove old particles
            if (particle.material.opacity <= 0) {
                this.scene.remove(particle);
                this.particles.splice(i, 1);
            }
        }

        this.renderer.render(this.scene, this.camera);
    }

    runSimulation() {
        if (!this.socket || !this.socket.connected) {
            console.error('Socket.IO not connected. Cannot send.');
            return;
        }

        const payload = {
            t0: 0,
            t1: 24,
            n: 1000,
            config: {
                "k_1": 0.5, "k_2": 0.1, "k_3": 0.2, "k_4": 0.3,
                "beta_1": 0.5, "beta_2": 0.5, "beta_3": 0.5, "beta_4": 0.5,
                "gamma_1": 0.5, "gamma_2": 0.5, "gamma_3": 0.5, "gamma_4": 0.5
            }
        };
        this.socket.emit('run', payload);
        console.log('Sent run request:', payload);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new WebGLDemo();
});
