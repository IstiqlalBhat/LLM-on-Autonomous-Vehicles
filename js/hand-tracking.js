/**
 * Hand Tracking State Manager
 * Manages hand detection state and grace periods
 */

export class HandTrackingState {
    constructor(gracePeriod) {
        this.isTracking = false;
        this.lastSeenTime = 0;
        this.gracePeriod = gracePeriod;
        this.transitioningOut = false;
    }
    
    updateTracking(handsDetected) {
        const now = Date.now();
        
        if (handsDetected) {
            this.isTracking = true;
            this.lastSeenTime = now;
            this.transitioningOut = false;
            return { status: 'tracking', timeSinceLost: 0 };
        }
        
        const timeSinceLost = now - this.lastSeenTime;
        
        if (timeSinceLost < this.gracePeriod) {
            // Within grace period - maintain last values
            this.transitioningOut = true;
            return { status: 'grace', timeSinceLost };
        }
        
        // Grace period expired
        this.isTracking = false;
        this.transitioningOut = false;
        return { status: 'lost', timeSinceLost };
    }
}

