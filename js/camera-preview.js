/**
 * Camera Preview Module
 * Handles camera feed rendering with hand landmarks
 */

export class CameraPreview {
    constructor(canvasId, width, height) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = width;
        this.canvas.height = height;
    }
    
    /**
     * Draw camera feed with hand landmarks
     * Uses global drawConnectors and drawLandmarks from MediaPipe
     */
    draw(results) {
        this.ctx.save();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(results.image, 0, 0, 
            this.canvas.width, this.canvas.height);
        
        if (results.multiHandLandmarks) {
            for (const landmarks of results.multiHandLandmarks) {
                // Draw hand connections
                window.drawConnectors(this.ctx, landmarks, window.HAND_CONNECTIONS, 
                    { color: 'rgba(0, 212, 255, 0.6)', lineWidth: 1 });
                // Draw hand landmarks
                window.drawLandmarks(this.ctx, landmarks, 
                    { color: 'rgba(255, 255, 255, 0.8)', lineWidth: 1, radius: 2 });
            }
        }
        this.ctx.restore();
    }
}

