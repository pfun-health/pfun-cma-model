/*
    fluid-demo.js
    A WebGL fluid simulation demo.
*/

class FluidDemo {
    constructor() {
        this.canvas = document.getElementById('fluid-canvas');
        this.gl = this.canvas.getContext('webgl');
        if (!this.gl) {
            console.error("WebGL not supported");
            return;
        }

        // Add floating point texture extension
        if (!this.gl.getExtension('OES_texture_float')) {
            console.error("Floating point textures not supported");
            return;
        }

        twgl.setDefaults({attribPrefix: "a_"});
        this.twgl = twgl.bind(this.gl);
        this.bufferInfo = this.twgl.createBufferInfoFromArrays({
            position: { data: [-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0], numComponents: 3 },
            indices: { data: [0, 1, 2, 2, 1, 3], numComponents: 2 },
        });

        this.initialize();
    }

    async initialize() {
        await this.setupShaders();
        this.setupBuffers();
        this.setupEventListeners();
        this.startAnimation();
    }

    async setupShaders() {
        const vs = `
            attribute vec4 a_position;
            varying vec2 v_texCoord;
            void main() {
                gl_Position = a_position;
                v_texCoord = a_position.xy * 0.5 + 0.5;
            }
        `;

        this.shaders = {
            advect: this.twgl.createProgramInfo(vs, `
                precision highp float;
                varying vec2 v_texCoord;
                uniform sampler2D u_velocityTexture;
                uniform sampler2D u_sourceTexture;
                uniform float u_dt;
                uniform vec2 u_resolution;

                void main() {
                    vec2 velocity = texture2D(u_velocityTexture, v_texCoord).xy;
                    vec2 coord = v_texCoord - velocity * u_dt / u_resolution;
                    gl_FragColor = texture2D(u_sourceTexture, coord);
                }
            `),
            divergence: this.twgl.createProgramInfo(vs, `
                precision highp float;
                varying vec2 v_texCoord;
                uniform sampler2D u_velocityTexture;
                uniform vec2 u_resolution;

                void main() {
                    float dx = 1.0 / u_resolution.x;
                    float dy = 1.0 / u_resolution.y;

                    float v_l = texture2D(u_velocityTexture, v_texCoord - vec2(dx, 0.0)).x;
                    float v_r = texture2D(u_velocityTexture, v_texCoord + vec2(dx, 0.0)).x;
                    float v_b = texture2D(u_velocityTexture, v_texCoord - vec2(0.0, dy)).y;
                    float v_t = texture2D(u_velocityTexture, v_texCoord + vec2(0.0, dy)).y;

                    gl_FragColor = vec4(0.5 * (v_r - v_l + v_t - v_b), 0.0, 0.0, 1.0);
                }
            `),
            jacobi: this.twgl.createProgramInfo(vs, `
                precision highp float;
                varying vec2 v_texCoord;
                uniform sampler2D u_pressureTexture;
                uniform sampler2D u_divergenceTexture;
                uniform vec2 u_resolution;

                void main() {
                    float dx = 1.0 / u_resolution.x;
                    float dy = 1.0 / u_resolution.y;

                    float p_l = texture2D(u_pressureTexture, v_texCoord - vec2(dx, 0.0)).x;
                    float p_r = texture2D(u_pressureTexture, v_texCoord + vec2(dx, 0.0)).x;
                    float p_b = texture2D(u_pressureTexture, v_texCoord - vec2(0.0, dy)).x;
                    float p_t = texture2D(u_pressureTexture, v_texCoord + vec2(0.0, dy)).x;

                    float divergence = texture2D(u_divergenceTexture, v_texCoord).x;

                    gl_FragColor = vec4(0.25 * (p_l + p_r + p_b + p_t - divergence), 0.0, 0.0, 1.0);
                }
            `),
            gradient: this.twgl.createProgramInfo(vs, `
                precision highp float;
                varying vec2 v_texCoord;
                uniform sampler2D u_pressureTexture;
                uniform sampler2D u_velocityTexture;
                uniform vec2 u_resolution;

                void main() {
                    float dx = 1.0 / u_resolution.x;
                    float dy = 1.0 / u_resolution.y;

                    float p_l = texture2D(u_pressureTexture, v_texCoord - vec2(dx, 0.0)).x;
                    float p_r = texture2D(u_pressureTexture, v_texCoord + vec2(dx, 0.0)).x;
                    float p_b = texture2D(u_pressureTexture, v_texCoord - vec2(0.0, dy)).x;
                    float p_t = texture2D(u_pressureTexture, v_texCoord + vec2(0.0, dy)).x;

                    vec2 velocity = texture2D(u_velocityTexture, v_texCoord).xy;
                    velocity -= 0.5 * vec2(p_r - p_l, p_t - p_b);

                    gl_FragColor = vec4(velocity, 0.0, 1.0);
                }
            `),
            display: this.twgl.createProgramInfo(vs, `
                precision highp float;
                varying vec2 v_texCoord;
                uniform sampler2D u_colorTexture;

                void main() {
                    gl_FragColor = texture2D(u_colorTexture, v_texCoord);
                }
            `),
             splat: this.twgl.createProgramInfo(vs, `
                precision highp float;
                varying vec2 v_texCoord;
                uniform sampler2D u_texture;
                uniform vec2 u_point;
                uniform vec3 u_color;
                uniform float u_radius;

                void main() {
                    float d = distance(v_texCoord, u_point);
                    if (d < u_radius) {
                        gl_FragColor = vec4(u_color, 1.0);
                    } else {
                        gl_FragColor = texture2D(u_texture, v_texCoord);
                    }
                }
            `),
        };
    }

    setupBuffers() {
        const attachments = [{ type: this.gl.FLOAT, min: this.gl.LINEAR, max: this.gl.LINEAR, wrap: this.gl.CLAMP_TO_EDGE }];
        this.velocityFbi1 = this.twgl.createFramebufferInfo(this.gl, attachments);
        this.velocityFbi2 = this.twgl.createFramebufferInfo(this.gl, attachments);
        this.colorFbi1 = this.twgl.createFramebufferInfo(this.gl, attachments);
        this.colorFbi2 = this.twgl.createFramebufferInfo(this.gl, attachments);
        this.pressureFbi1 = this.twgl.createFramebufferInfo(this.gl, attachments);
        this.pressureFbi2 = this.twgl.createFramebufferInfo(this.gl, attachments);
        this.divergenceFbi = this.twgl.createFramebufferInfo(this.gl, attachments);
    }

    setupEventListeners() {
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        window.addEventListener('resize', () => this.resize());
        this.resize();
    }

    resize() {
        this.twgl.resizeCanvasToDisplaySize(this.canvas);
        this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);

        // Resize framebuffers
        this.twgl.resizeFramebufferInfo(this.gl, this.velocityFbi1);
        this.twgl.resizeFramebufferInfo(this.gl, this.velocityFbi2);
        this.twgl.resizeFramebufferInfo(this.gl, this.colorFbi1);
        this.twgl.resizeFramebufferInfo(this.gl, this.colorFbi2);
        this.twgl.resizeFramebufferInfo(this.gl, this.pressureFbi1);
        this.twgl.resizeFramebufferInfo(this.gl, this.pressureFbi2);
        this.twgl.resizeFramebufferInfo(this.gl, this.divergenceFbi);
    }


    onMouseMove(e) {
        if (e.buttons) {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / this.canvas.clientWidth;
            const y = 1 - (e.clientY - rect.top) / this.canvas.clientHeight;

            this.splat(x, y, 0.01, [Math.random(), Math.random(), Math.random()]);
        }
    }

    splat(x, y, radius, color) {
        this.gl.useProgram(this.shaders.splat.program);
        this.twgl.setBuffersAndAttributes(this.gl, this.shaders.splat, this.bufferInfo);

        // Splat into color buffer
        this.twgl.bindFramebufferInfo(this.gl, this.colorFbi2);
        this.twgl.setUniforms(this.shaders.splat, {
            u_texture: this.colorFbi1.attachments[0],
            u_point: [x, y],
            u_radius: radius,
            u_color: color,
        });
        this.twgl.drawBufferInfo(this.gl, this.bufferInfo);
        this.swapColorBuffers();
    }

    startAnimation() {
        const animate = () => {
            this.updateSimulation();
            this.render();
            requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }

    updateSimulation() {
        const dt = 1 / 60;
        const resolution = [this.gl.canvas.width, this.gl.canvas.height];

        // Advect velocity
        this.gl.useProgram(this.shaders.advect.program);
        this.twgl.setBuffersAndAttributes(this.gl, this.shaders.advect, this.bufferInfo);
        this.twgl.bindFramebufferInfo(this.gl, this.velocityFbi2);
        this.twgl.setUniforms(this.shaders.advect, {
            u_velocityTexture: this.velocityFbi1.attachments[0],
            u_sourceTexture: this.velocityFbi1.attachments[0],
            u_dt: dt,
            u_resolution: resolution,
        });
        this.twgl.drawBufferInfo(this.gl, this.bufferInfo);
        this.swapVelocityBuffers();

        // Calculate divergence
        this.gl.useProgram(this.shaders.divergence.program);
        this.twgl.setBuffersAndAttributes(this.gl, this.shaders.divergence, this.bufferInfo);
        this.twgl.bindFramebufferInfo(this.gl, this.divergenceFbi);
        this.twgl.setUniforms(this.shaders.divergence, {
            u_velocityTexture: this.velocityFbi1.attachments[0],
            u_resolution: resolution,
        });
        this.twgl.drawBufferInfo(this.gl, this.bufferInfo);

        // Calculate pressure (Jacobi iteration)
        this.gl.useProgram(this.shaders.jacobi.program);
        this.twgl.setBuffersAndAttributes(this.gl, this.shaders.jacobi, this.bufferInfo);
        for (let i = 0; i < 20; i++) {
            this.twgl.bindFramebufferInfo(this.gl, this.pressureFbi2);
            this.twgl.setUniforms(this.shaders.jacobi, {
                u_pressureTexture: this.pressureFbi1.attachments[0],
                u_divergenceTexture: this.divergenceFbi.attachments[0],
                u_resolution: resolution,
            });
            this.twgl.drawBufferInfo(this.gl, this.bufferInfo);
            this.swapPressureBuffers();
        }

        // Subtract pressure gradient
        this.gl.useProgram(this.shaders.gradient.program);
        this.twgl.setBuffersAndAttributes(this.gl, this.shaders.gradient, this.bufferInfo);
        this.twgl.bindFramebufferInfo(this.gl, this.velocityFbi2);
        this.twgl.setUniforms(this.shaders.gradient, {
            u_pressureTexture: this.pressureFbi1.attachments[0],
            u_velocityTexture: this.velocityFbi1.attachments[0],
            u_resolution: resolution,
        });
        this.twgl.drawBufferInfo(this.gl, this.bufferInfo);
        this.swapVelocityBuffers();

        // Advect color
        this.gl.useProgram(this.shaders.advect.program);
        this.twgl.setBuffersAndAttributes(this.gl, this.shaders.advect, this.bufferInfo);
        this.twgl.bindFramebufferInfo(this.gl, this.colorFbi2);
        this.twgl.setUniforms(this.shaders.advect, {
            u_velocityTexture: this.velocityFbi1.attachments[0],
            u_sourceTexture: this.colorFbi1.attachments[0],
            u_dt: dt,
            u_resolution: resolution,
        });
        this.twgl.drawBufferInfo(this.gl, this.bufferInfo);
        this.swapColorBuffers();
    }

    render() {
        this.twgl.bindFramebufferInfo(this.gl, null); // Render to canvas
        this.gl.useProgram(this.shaders.display.program);
        this.twgl.setBuffersAndAttributes(this.gl, this.shaders.display, this.bufferInfo);
        this.twgl.setUniforms(this.shaders.display, {
            u_colorTexture: this.colorFbi1.attachments[0],
        });
        this.twgl.drawBufferInfo(this.gl, this.bufferInfo);
    }

    swapVelocityBuffers() {
        [this.velocityFbi1, this.velocityFbi2] = [this.velocityFbi2, this.velocityFbi1];
    }

    swapColorBuffers() {
        [this.colorFbi1, this.colorFbi2] = [this.colorFbi2, this.colorFbi1];
    }

    swapPressureBuffers() {
        [this.pressureFbi1, this.pressureFbi2] = [this.pressureFbi2, this.pressureFbi1];
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new FluidDemo();
});
