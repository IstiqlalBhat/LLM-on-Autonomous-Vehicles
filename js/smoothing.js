/**
 * Smoothing System
 * Provides classes for smooth value interpolation and filtering
 */

/**
 * Smooth value interpolation using critically damped spring
 */
export class SmoothValue {
    constructor(initialValue, smoothingFactor, deadZone) {
        this.current = initialValue;
        this.target = initialValue;
        this.velocity = 0;
        this.smoothingFactor = smoothingFactor;
        this.deadZone = deadZone;
        this.lastValidTarget = initialValue;
    }
    
    setTarget(value) {
        // Apply dead zone to prevent micro-jitter
        if (Math.abs(value - this.lastValidTarget) > this.deadZone) {
            this.target = value;
            this.lastValidTarget = value;
        }
    }
    
    update() {
        // Critically damped spring interpolation
        const diff = this.target - this.current;
        this.velocity += diff * this.smoothingFactor;
        this.velocity *= 0.7; // Damping
        this.current += this.velocity;
        return this.current;
    }
    
    updateSlow(defaultValue) {
        // Slower update for returning to defaults
        const diff = defaultValue - this.current;
        this.velocity += diff * 0.02;
        this.velocity *= 0.85;
        this.current += this.velocity;
        this.target = defaultValue;
        return this.current;
    }
}

/**
 * Exponential Moving Average filter for raw input
 */
export class EMAFilter {
    constructor(alpha) {
        this.alpha = alpha;
        this.value = null;
    }
    
    filter(newValue) {
        if (this.value === null) {
            this.value = newValue;
        } else {
            this.value = this.alpha * newValue + (1 - this.alpha) * this.value;
        }
        return this.value;
    }
    
    reset() {
        this.value = null;
    }
}

