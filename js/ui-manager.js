/**
 * UI Manager Module
 * Manages all UI updates and interactions
 */

export class UIManager {
    constructor() {
        this.elements = {
            statusDot: document.querySelector('.status-dot'),
            statusText: document.getElementById('status-text'),
            scaleBar: document.getElementById('scale-bar'),
            rotationXBar: document.getElementById('rotation-x-bar'),
            rotationYBar: document.getElementById('rotation-y-bar'),
            colorValue: document.getElementById('colorValue'),
            colorSwatches: document.querySelectorAll('.color-swatch'),
            loading: document.getElementById('loading')
        };
    }
    
    /**
     * Update hand tracking status display
     */
    updateStatus(className, text) {
        this.elements.statusDot.className = `status-dot ${className}`;
        this.elements.statusText.textContent = text;
    }
    
    /**
     * Update scale and rotation indicators
     */
    updateIndicators(scale, rotX, rotY) {
        // Scale bar
        const scalePercent = ((scale - 0.2) / 2.6) * 100;
        this.elements.scaleBar.style.width = 
            `${Math.min(Math.max(scalePercent, 5), 100)}%`;
        
        // Rotation X bar (centered, moving left/right)
        const rotXPercent = 50 + (rotX / Math.PI) * 40;
        this.elements.rotationXBar.style.left = 
            `${Math.min(Math.max(rotXPercent, 5), 95)}%`;
        
        // Rotation Y bar (centered, moving left/right)
        const rotYPercent = 50 + ((rotY % (Math.PI * 2)) / Math.PI) * 25;
        this.elements.rotationYBar.style.left = 
            `${Math.min(Math.max(rotYPercent, 5), 95)}%`;
    }
    
    /**
     * Hide loading screen
     */
    hideLoading() {
        this.elements.loading.classList.add('hidden');
    }
    
    /**
     * Update color value display
     */
    updateColorValue(color) {
        this.elements.colorValue.textContent = color.toUpperCase();
    }

    /**
     * Update active color swatch
     */
    updateActiveColorSwatch(color) {
        // Refresh swatches list in case DOM changed
        this.elements.colorSwatches = document.querySelectorAll('.color-swatch');
        
        // Remove active class from all swatches
        this.elements.colorSwatches.forEach(swatch => {
            swatch.classList.remove('active');
        });

        // Normalize colors for comparison
        const normalizedColor = color.toUpperCase();
        
        // Find and activate the swatch with matching color
        this.elements.colorSwatches.forEach(swatch => {
            const swatchColor = (swatch.dataset.color || '').toUpperCase();
            if (swatchColor === normalizedColor) {
                swatch.classList.add('active');
            }
        });
    }
}

