/**
 * Gesture Detection Module
 * Pure functions for detecting and analyzing hand gestures
 */

export const GestureDetection = {
    /**
     * Calculate hand orientation from landmarks
     */
    getHandOrientation(landmarks) {
        const wrist = landmarks[0];
        const middleMCP = landmarks[9];
        
        const palmVector = {
            x: middleMCP.x - wrist.x,
            y: middleMCP.y - wrist.y,
            z: middleMCP.z - wrist.z
        };
        
        // Tilt angle (rotation around X)
        const tiltX = Math.atan2(
            palmVector.y, 
            Math.sqrt(palmVector.x * palmVector.x + palmVector.z * palmVector.z)
        );
        
        // Rotation angle (rotation around Y)
        const tiltY = Math.atan2(palmVector.x, Math.abs(palmVector.z) + 0.001);
        
        return { tiltX, tiltY, palmVector };
    },
    
    /**
     * Check if hand is making a fist
     */
    isFist(landmarks) {
        const tips = [8, 12, 16, 20]; // Index, Middle, Ring, Pinky tips
        const mcps = [5, 9, 13, 17];  // Corresponding MCPs
        const wrist = landmarks[0];
        
        let closedFingers = 0;
        
        for (let i = 0; i < tips.length; i++) {
            const tip = landmarks[tips[i]];
            const mcp = landmarks[mcps[i]];
            
            const tipDist = Math.sqrt(
                Math.pow(tip.x - wrist.x, 2) + 
                Math.pow(tip.y - wrist.y, 2)
            );
            const mcpDist = Math.sqrt(
                Math.pow(mcp.x - wrist.x, 2) + 
                Math.pow(mcp.y - wrist.y, 2)
            );
            
            // If fingertip is closer to wrist than MCP, finger is closed
            if (tipDist < mcpDist * 1.15) {
                closedFingers++;
            }
        }
        
        return closedFingers >= 3;
    },
    
    /**
     * Calculate two-hand rotation (steering wheel gesture)
     */
    getTwoHandRotation(hand1, hand2) {
        const wrist1 = hand1[0];
        const wrist2 = hand2[0];
        
        // Angle between wrists determines Y rotation
        const angle = Math.atan2(
            wrist2.y - wrist1.y,
            wrist2.x - wrist1.x
        );
        
        // Average height determines X rotation
        const avgY = (wrist1.y + wrist2.y) / 2;
        
        return { 
            yRotation: angle,
            xRotation: (avgY - 0.5) * 2 // Map 0-1 to -1 to 1
        };
    },
    
    /**
     * Get pinch distance between thumb and index finger
     */
    getPinchDistance(landmarks) {
        const indexTip = landmarks[8];
        const thumbTip = landmarks[4];
        
        return Math.sqrt(
            Math.pow(indexTip.x - thumbTip.x, 2) + 
            Math.pow(indexTip.y - thumbTip.y, 2)
        );
    },
    
    /**
     * Get distance between two hands
     */
    getTwoHandDistance(hand1, hand2) {
        const palm1 = hand1[0];
        const palm2 = hand2[0];
        
        return Math.sqrt(
            Math.pow(palm1.x - palm2.x, 2) + 
            Math.pow(palm1.y - palm2.y, 2)
        );
    }
};

