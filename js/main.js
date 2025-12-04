/**
 * Main Entry Point
 * Initializes the application
 */

import { ParticleController } from './particle-controller.js';

// Wait for DOM to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

/**
 * Initialize the application
 */
function init() {
    // Create the particle controller instance
    const app = new ParticleController();
    
    // Set initial shape
    const initialBtn = document.querySelector('.shape-btn[data-shape="heart"]');
    app.setShape('heart', initialBtn);
    
    // Make app globally accessible for debugging (optional)
    window.particleApp = app;
    
    console.log('🎨 Particle Controller initialized');
}

