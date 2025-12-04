/**
 * Configuration and Constants
 * Central configuration for the particle system
 */

// Mobile device detection utility
export const DeviceDetector = {
    isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    },
    isTablet() {
        return /(iPad|tablet|playbook|silk)|(android(?!.*mobile))/i.test(navigator.userAgent);
    },
    isPhone() {
        return this.isMobile() && !this.isTablet();
    },
    isTouchDevice() {
        return ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
    },
    getDeviceType() {
        if (this.isPhone()) return 'phone';
        if (this.isTablet()) return 'tablet';
        return 'desktop';
    }
};

export const CONFIG = {
    particle: {
        count: 18000,
        size: 0.09, // Increased for better bloom visibility
        morphSpeed: 0.04,
        velocityDamping: 0.92
    },
    smoothing: {
        scale: { 
            factor: 0.12, 
            deadZone: 0.02 
        },
        rotation: { 
            factor: 0.08, 
            deadZone: 0.01 
        },
        emaAlpha: { 
            scale: 0.4, 
            rotation: 0.3, 
            pinch: 0.5 
        }
    },
    hand: {
        gracePeriod: 500,
        detectionConfidence: 0.7,
        trackingConfidence: 0.6,
        maxHands: 2
    },
    autoRotate: {
        speed: 0.003,
        oscillation: {
            amplitude: 0.08,
            frequency: 0.3
        }
    },
    repulsion: {
        radius: 2.0,      // Distance at which hand affects particles
        strength: 0.5,    // Force strength multiplier
        enabled: true     // Toggle repulsion on/off
    },
    gradient: {
        enabled: false,   // Toggle gradient mode
        mode: 'radial',   // 'radial', 'depth', 'velocity'
        baseHue: 0.6,     // Starting hue (0-1) - default blue
        hueRange: 0.3,    // How much the hue shifts (0-1)
        saturation: 0.8,  // Color saturation (0-1)
        lightness: 0.6,   // Color lightness (0-1)
        radius: 5.0       // Radius for radial gradient calculation
    }
};

// Mobile-optimized configuration
export function getOptimizedConfig() {
    const deviceType = DeviceDetector.getDeviceType();
    const config = JSON.parse(JSON.stringify(CONFIG)); // Deep clone

    if (deviceType === 'phone') {
        // Aggressive optimization for phones
        config.particle.count = 5000;
        config.particle.size = 0.11;
        config.hand.maxHands = 1;
        config.hand.detectionConfidence = 0.6;
        config.hand.trackingConfidence = 0.5;
    } else if (deviceType === 'tablet') {
        // Moderate optimization for tablets
        config.particle.count = 10000;
        config.particle.size = 0.10;
    }

    return config;
}

