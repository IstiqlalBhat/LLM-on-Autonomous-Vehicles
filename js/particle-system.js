/**
 * Particle System Module
 * Manages Three.js particle rendering and animation
 */

export class ParticleSystem {
    constructor(scene, config) {
        this.config = config;
        this.geometry = new THREE.BufferGeometry();
        this.positions = new Float32Array(config.count * 3);
        this.targetPositions = new Float32Array(config.count * 3);
        this.velocities = new Float32Array(config.count * 3);
        this.colors = new Float32Array(config.count * 3);

        // Initialize random positions
        for (let i = 0; i < config.count * 3; i++) {
            this.positions[i] = (Math.random() - 0.5) * 10;
            this.targetPositions[i] = this.positions[i];
            this.velocities[i] = 0;
        }

        // Initialize colors (default cyan)
        for (let i = 0; i < config.count * 3; i += 3) {
            this.colors[i] = 0.0;      // R
            this.colors[i + 1] = 0.83; // G
            this.colors[i + 2] = 1.0;  // B
        }

        this.geometry.setAttribute('position',
            new THREE.BufferAttribute(this.positions, 3));
        this.geometry.setAttribute('color',
            new THREE.BufferAttribute(this.colors, 3));

        const material = new THREE.PointsMaterial({
            color: 0xffffff, // White base so vertex colors show through
            size: config.size * 1.2, // Slightly larger for better bloom
            map: this.createTexture(),
            transparent: true,
            opacity: 1.0, // Full opacity for brighter bloom
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            sizeAttenuation: true, // Particles get smaller with distance
            vertexColors: true // Enable vertex colors
        });

        this.mesh = new THREE.Points(this.geometry, material);
        this.baseColor = new THREE.Color(0x00d4ff); // Store base color
        scene.add(this.mesh);
    }
    
    /**
     * Create soft glowing texture for particles
     */
    createTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);

        // Enhanced gradient for better bloom effect
        grad.addColorStop(0, 'rgba(255,255,255,1)');    // Bright white core
        grad.addColorStop(0.2, 'rgba(255,255,255,0.9)'); // Strong inner glow
        grad.addColorStop(0.5, 'rgba(255,255,255,0.4)'); // Mid glow
        grad.addColorStop(0.8, 'rgba(255,255,255,0.1)'); // Soft outer glow
        grad.addColorStop(1, 'rgba(255,255,255,0)');     // Fade to transparent

        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 64, 64);
        const texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;
        return texture;
    }
    
    /**
     * Set target positions for particle morphing
     */
    setTargetPositions(positions) {
        for (let i = 0; i < this.targetPositions.length; i++) {
            this.targetPositions[i] = positions[i];
        }
    }
    
    /**
     * Update particle positions with velocity-based morphing and hand repulsion
     */
    update(handPosition = null, repulsionConfig = null, gradientConfig = null) {
        const { morphSpeed, velocityDamping } = this.config;
        const posAttr = this.geometry.attributes.position;
        const array = posAttr.array;

        for (let i = 0; i < this.config.count; i++) {
            const i3 = i * 3;

            // Morphing logic
            const diffX = this.targetPositions[i3] - array[i3];
            const diffY = this.targetPositions[i3 + 1] - array[i3 + 1];
            const diffZ = this.targetPositions[i3 + 2] - array[i3 + 2];

            this.velocities[i3] += diffX * morphSpeed;
            this.velocities[i3 + 1] += diffY * morphSpeed;
            this.velocities[i3 + 2] += diffZ * morphSpeed;

            this.velocities[i3] *= velocityDamping;
            this.velocities[i3 + 1] *= velocityDamping;
            this.velocities[i3 + 2] *= velocityDamping;

            array[i3] += this.velocities[i3];
            array[i3 + 1] += this.velocities[i3 + 1];
            array[i3 + 2] += this.velocities[i3 + 2];

            // Hand repulsion logic
            if (handPosition && repulsionConfig && repulsionConfig.enabled) {
                const dx = array[i3] - handPosition.x;
                const dy = array[i3 + 1] - handPosition.y;
                const dz = array[i3 + 2] - handPosition.z;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < repulsionConfig.radius && dist > 0.01) {
                    const force = (repulsionConfig.radius - dist) / repulsionConfig.radius;
                    const repulsionForce = force * repulsionConfig.strength;

                    array[i3] += (dx / dist) * repulsionForce;
                    array[i3 + 1] += (dy / dist) * repulsionForce;
                    array[i3 + 2] += (dz / dist) * repulsionForce;
                }
            }
        }

        posAttr.needsUpdate = true;

        // Update colors with gradient if enabled
        if (gradientConfig && gradientConfig.enabled) {
            this.updateGradientColors(gradientConfig);
        }
    }

    /**
     * Update particle colors based on gradient configuration
     */
    updateGradientColors(gradientConfig) {
        const colorAttr = this.geometry.attributes.color;
        const colors = colorAttr.array;
        const positions = this.geometry.attributes.position.array;
        const color = new THREE.Color();
        const time = Date.now() * 0.001; // Add time-based animation

        for (let i = 0; i < this.config.count; i++) {
            const i3 = i * 3;
            const x = positions[i3];
            const y = positions[i3 + 1];
            const z = positions[i3 + 2];

            let hue;
            let saturation = gradientConfig.saturation || 0.9;
            let lightness = gradientConfig.lightness || 0.6;

            switch (gradientConfig.mode) {
                case 'radial':
                    // Color based on distance from center with animated pulse
                    const distFromCenter = Math.sqrt(x * x + y * y + z * z);
                    const normalizedDist = Math.min(distFromCenter / gradientConfig.radius, 1.0);
                    // Add subtle animation
                    const pulse = Math.sin(time * 0.5 + distFromCenter) * 0.05;
                    hue = gradientConfig.baseHue + (normalizedDist * gradientConfig.hueRange) + pulse;
                    // Brighter at center, dimmer at edges
                    lightness = 0.65 - (normalizedDist * 0.15);
                    break;

                case 'depth':
                    // Color based on Z position with wave effect
                    const normalizedZ = (z + 5) / 10;
                    const wave = Math.sin(time + y * 0.5) * 0.05;
                    hue = gradientConfig.baseHue + (normalizedZ * gradientConfig.hueRange) + wave;
                    // Vary lightness based on depth
                    lightness = 0.5 + (normalizedZ * 0.2);
                    break;

                case 'velocity':
                    // Color based on velocity magnitude - more dramatic effect
                    const velMag = Math.sqrt(
                        this.velocities[i3] * this.velocities[i3] +
                        this.velocities[i3 + 1] * this.velocities[i3 + 1] +
                        this.velocities[i3 + 2] * this.velocities[i3 + 2]
                    );
                    // More sensitive velocity detection
                    const normalizedVel = Math.min(velMag * 20, 1.0);
                    hue = gradientConfig.baseHue + (normalizedVel * gradientConfig.hueRange);
                    // Faster particles are brighter
                    lightness = 0.5 + (normalizedVel * 0.25);
                    saturation = 0.8 + (normalizedVel * 0.2);
                    break;

                default:
                    hue = gradientConfig.baseHue;
            }

            // Wrap hue to 0-1 range (handle negative values too)
            hue = ((hue % 1.0) + 1.0) % 1.0;

            // Set color using HSL with enhanced values
            color.setHSL(hue, Math.min(saturation, 1.0), Math.min(lightness, 0.75));

            colors[i3] = color.r;
            colors[i3 + 1] = color.g;
            colors[i3 + 2] = color.b;
        }

        colorAttr.needsUpdate = true;
    }
    
    /**
     * Set particle color (solid color mode)
     */
    setColor(color) {
        if (!color) return;

        try {
            // Create Three.js Color object from hex string
            const threeColor = new THREE.Color(color);
            
            // Store base color
            this.baseColor.copy(threeColor);

            // Set all particles to the same color (solid mode)
            const colorAttr = this.geometry.attributes.color;
            if (!colorAttr) {
                console.error('Color attribute not found in geometry');
                return;
            }
            
            const colors = colorAttr.array;

            for (let i = 0; i < this.config.count; i++) {
                const i3 = i * 3;
                colors[i3] = this.baseColor.r;
                colors[i3 + 1] = this.baseColor.g;
                colors[i3 + 2] = this.baseColor.b;
            }

            colorAttr.needsUpdate = true;
        } catch (error) {
            console.error('Error setting particle color:', error, color);
        }
    }
    
    /**
     * Set particle system transform (scale and rotation)
     */
    setTransform(scale, rotX, rotY) {
        this.mesh.scale.set(scale, scale, scale);
        this.mesh.rotation.x = rotX;
        this.mesh.rotation.y = rotY;
    }
}

