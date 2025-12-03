/**
 * Configuration and Constants
 * Central configuration for the particle system
 */

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

