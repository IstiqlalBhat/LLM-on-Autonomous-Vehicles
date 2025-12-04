/**
 * Main Particle Controller
 * Orchestrates all modules and manages the application
 */

import { CONFIG, getOptimizedConfig, DeviceDetector } from './config.js';
import { SmoothValue, EMAFilter } from './smoothing.js';
import { HandTrackingState } from './hand-tracking.js';
import { GestureDetection } from './gesture-detection.js';
import { ShapeGenerator } from './shape-generator.js';
import { ParticleSystem } from './particle-system.js';
import { UIManager } from './ui-manager.js';
import { CameraPreview } from './camera-preview.js';

export class ParticleController {
    constructor() {
        // Detect device and apply optimized config
        this.isMobile = DeviceDetector.isMobile();
        this.deviceType = DeviceDetector.getDeviceType();
        this.config = getOptimizedConfig();

        this.gestureMode = 'both';
        this.currentShape = 'heart';
        this.autoRotate = true;
        this.baseRotationY = 0;
        this.handPosition3D = null; // 3D position of hand in world space

        console.log(`Device detected: ${this.deviceType}`, {
            particleCount: this.config.particle.count,
            isMobile: this.isMobile
        });

        this.initializeSmoothing();
        this.initializeThreeJS();
        this.initializeUI();
        this.initializeHandTracking();
        this.setupEventListeners();
        this.initializeColorPicker();
        this.animate();
    }

    /**
     * Initialize color picker with default color
     */
    initializeColorPicker() {
        const defaultColor = '#00d4ff';
        const defaultName = 'Cyan';
        this.setParticleColor(defaultColor, defaultName);
    }
    
    /**
     * Initialize smoothing system
     */
    initializeSmoothing() {
        const { scale, rotation, emaAlpha } = this.config.smoothing;
        
        // Smooth values for scale and rotation
        this.smoothScale = new SmoothValue(1.0, scale.factor, scale.deadZone);
        this.smoothRotationX = new SmoothValue(0, rotation.factor, rotation.deadZone);
        this.smoothRotationY = new SmoothValue(0, rotation.factor, rotation.deadZone);
        
        // EMA filters for raw input
        this.scaleFilter = new EMAFilter(emaAlpha.scale);
        this.rotXFilter = new EMAFilter(emaAlpha.rotation);
        this.rotYFilter = new EMAFilter(emaAlpha.rotation);
        this.pinchFilter = new EMAFilter(emaAlpha.pinch);
        
        // Hand tracking state
        this.handState = new HandTrackingState(this.config.hand.gracePeriod);
    }
    
    /**
     * Initialize Three.js scene, camera, and renderer
     */
    initializeThreeJS() {
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x090a0f, 0.015);

        this.camera = new THREE.PerspectiveCamera(
            75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.z = 8;

        this.renderer = new THREE.WebGLRenderer({
            antialias: !this.isMobile, // Disable antialiasing on mobile for performance
            alpha: true,
            powerPreference: this.isMobile ? 'low-power' : 'high-performance'
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);

        // Limit pixel ratio on mobile devices for better performance
        const maxPixelRatio = this.deviceType === 'phone' ? 1.5 : 2;
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, maxPixelRatio));
        this.renderer.toneMapping = THREE.ReinhardToneMapping;
        document.body.appendChild(this.renderer.domElement);

        this.particles = new ParticleSystem(this.scene, this.config.particle);

        // Setup Bloom Post-Processing
        this.setupBloom();
    }

    /**
     * Setup bloom post-processing effect
     */
    setupBloom() {
        // Create effect composer
        this.composer = new THREE.EffectComposer(this.renderer);

        // Add render pass (renders the scene)
        const renderPass = new THREE.RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        // Adjust bloom settings based on device type
        let bloomStrength, bloomRadius, bloomThreshold;
        if (this.deviceType === 'phone') {
            bloomStrength = 1.0;  // Reduced for performance
            bloomRadius = 0.3;
            bloomThreshold = 0.9;
        } else if (this.deviceType === 'tablet') {
            bloomStrength = 1.2;
            bloomRadius = 0.35;
            bloomThreshold = 0.87;
        } else {
            bloomStrength = 1.5;
            bloomRadius = 0.4;
            bloomThreshold = 0.85;
        }

        // Add bloom pass (creates the glow effect)
        const bloomPass = new THREE.UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            bloomStrength,
            bloomRadius,
            bloomThreshold
        );
        this.composer.addPass(bloomPass);

        // Store bloom pass for potential adjustments
        this.bloomPass = bloomPass;
    }
    
    /**
     * Initialize UI managers
     */
    initializeUI() {
        this.uiManager = new UIManager();
        this.cameraPreview = new CameraPreview('preview-canvas', 200, 150);
    }
    
    /**
     * Initialize MediaPipe hand tracking
     */
    initializeHandTracking() {
        const videoElement = document.querySelector('.input_video');
        
        this.hands = new window.Hands({
            locateFile: (file) => 
                `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        
        // Adjust MediaPipe model complexity based on device
        const modelComplexity = this.deviceType === 'phone' ? 0 : 1;

        this.hands.setOptions({
            maxNumHands: this.config.hand.maxHands,
            modelComplexity: modelComplexity,
            minDetectionConfidence: this.config.hand.detectionConfidence,
            minTrackingConfidence: this.config.hand.trackingConfidence
        });

        this.hands.onResults(this.onHandsResults.bind(this));

        // Reduce camera resolution on mobile devices
        const cameraWidth = this.deviceType === 'phone' ? 480 : 640;
        const cameraHeight = this.deviceType === 'phone' ? 360 : 480;

        this.cameraFeed = new window.Camera(videoElement, {
            onFrame: async () => {
                await this.hands.send({ image: videoElement });
            },
            width: cameraWidth,
            height: cameraHeight,
            facingMode: this.isMobile ? 'user' : undefined // Ensure front camera on mobile
        });
        this.cameraFeed.start();
    }
    
    /**
     * Setup event listeners for UI interactions
     */
    setupEventListeners() {
        // Mobile UI toggle button
        const toggleBtn = document.getElementById('ui-toggle');
        const uiContainer = document.getElementById('ui-container');
        if (toggleBtn && this.isMobile) {
            toggleBtn.addEventListener('click', () => {
                uiContainer.classList.toggle('collapsed');
            });
        }

        // Shape buttons
        document.querySelectorAll('.shape-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.setShape(btn.dataset.shape, btn);
            });
        });
        
        // Mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-btn').forEach(b => 
                    b.classList.remove('active'));
                btn.classList.add('active');
                this.gestureMode = btn.dataset.mode;
            });
        });
        
        // Color selector dropdown
        this.setupColorSelector();

        // Gradient toggle
        const gradientToggle = document.getElementById('gradientToggle');
        const gradientControls = document.getElementById('gradientControls');
        if (gradientToggle) {
            gradientToggle.addEventListener('change', (e) => {
                this.config.gradient.enabled = e.target.checked;

                // Show/hide gradient controls with animation
                if (gradientControls) {
                    if (e.target.checked) {
                        gradientControls.style.display = 'block';
                        gradientControls.style.opacity = '0';
                        requestAnimationFrame(() => {
                            gradientControls.style.opacity = '1';
                        });
                    } else {
                        gradientControls.style.opacity = '0';
                        setTimeout(() => {
                            gradientControls.style.display = 'none';
                        }, 200);
                    }
                }

                // If disabling gradient, reapply current solid color
                if (!e.target.checked) {
                    const currentColor = '#' + this.particles.baseColor.getHexString().toUpperCase();
                    this.particles.setColor(currentColor);
                }
            });
        }

        // Gradient mode buttons
        document.querySelectorAll('.gradient-mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.gradient-mode-btn').forEach(b =>
                    b.classList.remove('active'));
                btn.classList.add('active');
                this.config.gradient.mode = btn.dataset.gradientMode;
            });
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.composer.setSize(window.innerWidth, window.innerHeight);
        });
    }
    
    /**
     * Convert 2D hand position to 3D world coordinates
     */
    screenToWorld(normalizedX, normalizedY) {
        // Convert normalized coordinates (0-1) to NDC (-1 to 1)
        const x = (normalizedX * 2) - 1;
        const y = -(normalizedY * 2) + 1; // Flip Y axis

        // Create a vector at the near plane
        const vector = new THREE.Vector3(x, y, 0.5);

        // Unproject to get world coordinates
        vector.unproject(this.camera);

        // Calculate direction from camera to the point
        const dir = vector.sub(this.camera.position).normalize();

        // Calculate distance to place hand at same depth as particles (z=0 plane)
        const distance = -this.camera.position.z / dir.z;

        // Calculate final position
        const pos = this.camera.position.clone().add(dir.multiplyScalar(distance));

        return pos;
    }

    /**
     * Set particle shape
     */
    setShape(type, buttonElement) {
        this.currentShape = type;
        const positions = ShapeGenerator.generate(type, this.config.particle.count);
        this.particles.setTargetPositions(positions);

        document.querySelectorAll('.shape-btn').forEach(b =>
            b.classList.remove('active'));
        if (buttonElement) {
            buttonElement.classList.add('active');
        }
    }

    /**
     * Setup color selector - simple grid version
     */
    setupColorSelector() {
        const colorBtns = document.querySelectorAll('.color-btn');
        const colorPicker = document.getElementById('colorPicker');
        const colorValueDisplay = document.getElementById('colorValue');
        
        // Color button clicks
        colorBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const color = btn.dataset.color;
                if (color) {
                    // Update active state
                    colorBtns.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    
                    // Set color
                    this.setParticleColor(color, btn.title);
                    
                    // Update picker and display
                    if (colorPicker) colorPicker.value = color;
                    if (colorValueDisplay) colorValueDisplay.textContent = color.toUpperCase();
                }
            });
        });
        
        // Custom color picker
        if (colorPicker) {
            colorPicker.addEventListener('input', (e) => {
                const color = e.target.value.toUpperCase();
                this.setParticleColor(color, 'Custom');
                
                // Update display
                if (colorValueDisplay) colorValueDisplay.textContent = color;
                
                // Remove active from preset buttons
                colorBtns.forEach(b => b.classList.remove('active'));
            });
        }
    }
    
    /**
     * Set particle color
     */
    setParticleColor(color, name = null) {
        if (!color) return;
        
        // Normalize color to uppercase for consistency
        const normalizedColor = color.toUpperCase();
        
        // Ensure it starts with #
        const hexColor = normalizedColor.startsWith('#') ? normalizedColor : '#' + normalizedColor;
        
        // Update particle material color
        this.particles.setColor(hexColor);
        
        // Also update gradient base hue from the selected color
        const threeColor = new THREE.Color(hexColor);
        const hsl = { h: 0, s: 0, l: 0 };
        threeColor.getHSL(hsl);
        this.config.gradient.baseHue = hsl.h;
        this.config.gradient.saturation = Math.max(hsl.s, 0.7); // Keep saturation high
        this.config.gradient.lightness = Math.max(hsl.l, 0.5);  // Keep lightness reasonable
        
        // Update UI
        this.updateColorUI(hexColor, name);
        
        // Update custom color picker and input
        const colorPicker = document.getElementById('colorPicker');
        const colorInput = document.getElementById('colorInput');
        if (colorPicker) colorPicker.value = hexColor;
        if (colorInput) colorInput.value = hexColor;
    }
    
    /**
     * Update color UI elements
     */
    updateColorUI(color, name = null) {
        const colorPreview = document.getElementById('colorPreview');
        const colorNameEl = document.getElementById('colorName');
        const colorValueEl = document.getElementById('colorValue');
        const colorOptions = document.querySelectorAll('.color-option');
        
        // Update preview
        if (colorPreview) {
            colorPreview.style.background = color;
        }
        
        // Update name
        if (colorNameEl) {
            colorNameEl.textContent = name || 'Custom';
        }
        
        // Update hex value
        if (colorValueEl) {
            colorValueEl.textContent = color;
        }
        
        // Update active option
        colorOptions.forEach(option => {
            option.classList.remove('active');
            if (option.dataset.color && option.dataset.color.toUpperCase() === color.toUpperCase()) {
                option.classList.add('active');
                // Update name if not provided
                if (!name && colorNameEl) {
                    colorNameEl.textContent = option.dataset.name || 'Custom';
                }
            }
        });
    }
    
    /**
     * Handle hand tracking results from MediaPipe
     */
    onHandsResults(results) {
        this.uiManager.hideLoading();
        this.cameraPreview.draw(results);
        
        const trackingStatus = this.handState.updateTracking(
            results.multiHandLandmarks && results.multiHandLandmarks.length > 0);
        
        if (trackingStatus.status === 'tracking') {
            this.processGestures(results.multiHandLandmarks);
        } else if (trackingStatus.status === 'lost') {
            this.handleNoHands(trackingStatus.timeSinceLost);
        }
    }
    
    /**
     * Process detected gestures
     */
    processGestures(hands) {
        if (hands.length >= 2) {
            this.processTwoHandGestures(hands[0], hands[1]);
        } else {
            this.processSingleHandGestures(hands[0]);
        }

        // Update hand position for repulsion (use palm position - landmark 0)
        if (hands.length > 0) {
            const palm = hands[0][0]; // Wrist/palm base position
            this.handPosition3D = this.screenToWorld(palm.x, palm.y);
        } else {
            this.handPosition3D = null;
        }
    }
    
    /**
     * Process two-hand gestures (scale and rotation)
     */
    processTwoHandGestures(hand1, hand2) {
        // Scale: Distance between hands
        if (this.gestureMode === 'scale' || this.gestureMode === 'both') {
            const distance = GestureDetection.getTwoHandDistance(hand1, hand2);
            const filtered = this.scaleFilter.filter(distance);
            let scale = 0.3 + (filtered * 3.5);
            scale = Math.min(Math.max(scale, 0.2), 2.8);
            this.smoothScale.setTarget(scale);
        }
        
        // Rotation: Two-hand steering gesture
        if (this.gestureMode === 'rotate' || this.gestureMode === 'both') {
            const rot = GestureDetection.getTwoHandRotation(hand1, hand2);
            this.smoothRotationY.setTarget(this.rotYFilter.filter(rot.yRotation * 2));
            this.smoothRotationX.setTarget(this.rotXFilter.filter(rot.xRotation * 0.8));
            this.autoRotate = false;
        }
        
        this.uiManager.updateStatus('rotate', 'Two hands: Steering control');
    }
    
    /**
     * Process single-hand gestures
     */
    processSingleHandGestures(hand) {
        if (GestureDetection.isFist(hand)) {
            this.autoRotate = false;
            this.uiManager.updateStatus('pinch', 'Fist: Rotation locked');
            return;
        }
        
        // Scale: Pinch gesture
        if (this.gestureMode === 'scale' || this.gestureMode === 'both') {
            const pinch = GestureDetection.getPinchDistance(hand);
            const filtered = this.pinchFilter.filter(pinch);
            let scale = 0.3 + (filtered * 8);
            scale = Math.min(Math.max(scale, 0.2), 2.5);
            this.smoothScale.setTarget(scale);
        }
        
        // Rotation: Hand tilt
        if (this.gestureMode === 'rotate' || this.gestureMode === 'both') {
            const orientation = GestureDetection.getHandOrientation(hand);
            this.smoothRotationX.setTarget(this.rotXFilter.filter(orientation.tiltX * 1.2));
            this.smoothRotationY.setTarget(this.rotYFilter.filter(orientation.tiltY * 0.8));
            this.autoRotate = false;
        }
        
        // Update UI status based on pinch distance
        const pinchDist = GestureDetection.getPinchDistance(hand);
        const statusClass = pinchDist < 0.05 ? 'pinch' : 'active';
        const statusText = pinchDist < 0.05 ? 'Pinching (Compress)' :
            this.gestureMode === 'rotate' ? 'Tilt to rotate' :
            this.gestureMode === 'both' ? 'Tilt + Pinch active' :
            'Open hand (Expand)';
        
        this.uiManager.updateStatus(statusClass, statusText);
    }
    
    /**
     * Handle when no hands are detected
     */
    handleNoHands(timeSinceLost) {
        // Reset filters after grace period
        if (timeSinceLost > this.config.hand.gracePeriod + 500) {
            this.scaleFilter.reset();
            this.rotXFilter.reset();
            this.rotYFilter.reset();
            this.pinchFilter.reset();
        }

        this.handPosition3D = null; // Clear hand position
        this.autoRotate = true;
        this.uiManager.updateStatus('', 'No hands detected');
    }
    
    /**
     * Main animation loop
     */
    animate() {
        requestAnimationFrame(this.animate.bind(this));
        
        const trackingStatus = this.handState.updateTracking(false);
        let currentScale, currentRotX, currentRotY;
        
        // Update smooth values based on tracking state
        if (trackingStatus.status === 'tracking' || 
            trackingStatus.status === 'grace') {
            // Active tracking or grace period
            currentScale = this.smoothScale.update();
            currentRotX = this.smoothRotationX.update();
            currentRotY = this.smoothRotationY.update();
        } else {
            // No hands - return to defaults
            currentScale = this.smoothScale.updateSlow(1.0);
            
            if (this.autoRotate) {
                this.baseRotationY += this.config.autoRotate.speed;
                const time = Date.now() * 0.001;
                this.smoothRotationY.target = this.baseRotationY;
                this.smoothRotationX.target = 
                    Math.sin(time * this.config.autoRotate.oscillation.frequency) * 
                    this.config.autoRotate.oscillation.amplitude;
                currentRotX = this.smoothRotationX.update();
                currentRotY = this.smoothRotationY.update();
            } else {
                currentRotX = this.smoothRotationX.updateSlow(0);
                currentRotY = this.smoothRotationY.updateSlow(this.baseRotationY);
            }
        }
        
        // Update particles with hand repulsion and gradient
        this.particles.update(this.handPosition3D, this.config.repulsion, this.config.gradient);
        this.particles.setTransform(currentScale, currentRotX, currentRotY);
        
        // Update UI
        this.uiManager.updateIndicators(currentScale, currentRotX, currentRotY);

        // Render scene with bloom effect
        this.composer.render();
    }
}

