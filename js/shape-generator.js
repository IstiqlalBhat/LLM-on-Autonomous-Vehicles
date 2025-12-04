/**
 * Shape Generator Module
 * Generates particle positions for various 3D shapes
 */

export const ShapeGenerator = {
    /**
     * Generate random point on sphere surface
     */
    getPointOnSphere(radius = 3) {
        const u = Math.random();
        const v = Math.random();
        const theta = 2 * Math.PI * u;
        const phi = Math.acos(2 * v - 1);
        return {
            x: radius * Math.sin(phi) * Math.cos(theta),
            y: radius * Math.sin(phi) * Math.sin(theta),
            z: radius * Math.cos(phi)
        };
    },
    
    /**
     * Generate heart shape
     */
    heart() {
        const t = Math.random() * Math.PI * 2;
        const u = Math.random() * Math.PI;
        const scale = 0.16;
        const x = 16 * Math.pow(Math.sin(t), 3) * scale;
        const y = (13 * Math.cos(t) - 5 * Math.cos(2*t) - 
                  2 * Math.cos(3*t) - Math.cos(4*t)) * scale + 0.5;
        const z = 4 * Math.cos(u) * Math.sin(t) * scale;
        return { x, y, z };
    },
    
    /**
     * Generate Saturn (planet with rings)
     */
    saturn() {
        const r = Math.random();
        if (r > 0.35) {
            // Planet body
            return this.getPointOnSphere(1.8);
        }
        // Rings
        const angle = Math.random() * Math.PI * 2;
        const dist = 2.8 + Math.random() * 1.8;
        let x = Math.cos(angle) * dist;
        let z = Math.sin(angle) * dist;
        let y = (Math.random() - 0.5) * 0.15;
        
        // Tilt the ring
        const tilt = 0.4;
        const newY = y * Math.cos(tilt) - z * Math.sin(tilt);
        const newZ = y * Math.sin(tilt) + z * Math.cos(tilt);
        return { x, y: newY, z: newZ };
    },
    
    /**
     * Generate flower shape
     */
    flower() {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;
        const petals = 6;
        const r = 2.2 + Math.sin(petals * theta) * Math.sin(petals * phi) * 0.8;
        return {
            x: r * Math.sin(phi) * Math.cos(theta),
            y: r * Math.sin(phi) * Math.sin(theta),
            z: r * Math.cos(phi) * 0.6
        };
    },
    
    /**
     * Generate Buddha statue (torus knot)
     */
    buddha() {
        const u = Math.random() * Math.PI * 2;
        const v = Math.random() * Math.PI * 2;
        const tube = 0.7;
        const rad = 2.2;
        const p = 2, q = 3;
        const r = rad + tube * Math.cos(q * u);
        let x = r * Math.cos(p * u);
        let y = r * Math.sin(p * u);
        let z = tube * Math.sin(q * u);
        
        // Add some random fill
        if (Math.random() > 0.75) {
            const scale = Math.random();
            x *= scale; y *= scale; z *= scale;
        }
        return { x, y, z };
    },
    
    /**
     * Generate fireworks burst pattern
     */
    fireworks() {
        const burstCenters = [
            { x: 0, y: 0, z: 0 },
            { x: -2.5, y: 1.5, z: -1 },
            { x: 2.5, y: 1, z: 0.5 },
            { x: 0, y: -2, z: 1 }
        ];
        const burst = burstCenters[Math.floor(Math.random() * burstCenters.length)];
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;
        const r = Math.random() * 2.5;
        return {
            x: burst.x + r * Math.sin(phi) * Math.cos(theta),
            y: burst.y + r * Math.sin(phi) * Math.sin(theta),
            z: burst.z + r * Math.cos(phi)
        };
    },
    
    /**
     * Generate sphere
     */
    sphere() {
        return this.getPointOnSphere(3);
    },
    
    /**
     * Generate array of positions for a given shape type
     */
    generate(type, count) {
        const positions = [];
        for (let i = 0; i < count; i++) {
            const point = this[type] ? this[type]() : this.sphere();
            positions.push(point.x, point.y, point.z);
        }
        return positions;
    }
};

