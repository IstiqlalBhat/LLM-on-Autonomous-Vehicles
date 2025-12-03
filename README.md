# Gesture-Controlled Particle System

Wave your hands and watch thousands of glowing particles respond. This project uses your webcam to track hand gestures and lets you sculpt, spin, and play with 3D particle formations in real-time.

## What it does

Control 18,000 particles with your bare hands. No mouse, no keyboard - just gestures:

- **Pinch** to compress particles into a tight cluster
- **Spread your hands** to expand them outward
- **Tilt** to rotate the formation
- **Move through the cloud** and particles scatter away from your hand (force field effect)
- **Switch colors** - 20+ presets or pick your own
- **Enable gradients** for rainbow effects that shift with particle depth, position, or velocity
- **Morph between shapes** - heart, flower, saturn, buddha statue, fireworks, sphere

The whole thing runs at 60fps with bloom post-processing that makes particles glow like energy orbs.

## Running it

You need a web server (ES6 modules won't work with `file://` URLs). Pick one:

```bash
# Python (easiest)
python -m http.server 8000

# Node
npx http-server -p 8000

# PHP
php -S localhost:8000
```

Then open `http://localhost:8000` and allow camera access when prompted.

**Note**: Make sure you have decent lighting - MediaPipe needs to see your hands clearly.

## How to use it

**One hand gestures:**
- Pinch (thumb + index finger close together) = shrink particles
- Open hand = expand particles
- Tilt your hand = rotate the formation
- Move your palm through the cloud = particles scatter away from you

**Two hands:**
- Move hands closer/farther apart = scale up/down
- Tilt both hands together = steer the rotation

**Fist** = locks rotation in place

**UI panel (left side):**
- Click shape buttons to morph between formations
- Toggle "Dynamic Gradient" for color effects:
  - **Radial**: colors shift from center → edge
  - **Depth**: colors based on front/back position
  - **Velocity**: faster-moving particles change color
- Pick colors from the grid or use custom picker
- Switch gesture modes (scale only, rotate only, or both)

## How it works

The codebase is split into focused modules:

- **particle-controller.js** - main orchestrator, coordinates everything
- **particle-system.js** - Three.js rendering, handles 18k particles, repulsion physics, and gradient calculations
- **gesture-detection.js** - pure functions that analyze hand positions (pinch distance, tilt angles, etc.)
- **hand-tracking.js** - manages MediaPipe state, handles "grace periods" when hands briefly leave frame
- **smoothing.js** - critically damped springs and EMA filters for smooth motion (no jitter)
- **shape-generator.js** - math functions that generate coordinates for each shape
- **config.js** - tune everything from one place (particle count, smoothing factors, repulsion strength, etc.)

**Tech stack:**
- Three.js r128 for 3D rendering
- Three.js post-processing (EffectComposer + UnrealBloomPass) for glow effects
- MediaPipe Hands for webcam tracking
- Vanilla JS with ES6 modules (no framework bloat)

## Customizing

Everything's configurable in `js/config.js`. Some things you might want to tweak:

**Particle count:**
```javascript
particle: {
    count: 18000,  // More = prettier but slower. 10k-25k is the sweet spot
}
```

**Hand repulsion strength:**
```javascript
repulsion: {
    radius: 2.0,    // How far the force field extends
    strength: 0.5,  // How hard particles get pushed (0.3-0.8 recommended)
}
```

**Gradient settings:**
```javascript
gradient: {
    baseHue: 0.6,      // Starting color (0=red, 0.33=green, 0.66=blue)
    hueRange: 0.3,     // How much colors shift (0.2-0.5 looks good)
    saturation: 0.8,   // Color intensity
}
```

**Smoothing (if gestures feel too twitchy or laggy):**
```javascript
smoothing: {
    scale: { factor: 0.12 },     // Higher = snappier response
    rotation: { factor: 0.08 },  // Lower = smoother but slower
}
```

**Adding your own shapes:**

1. Write a function in `js/shape-generator.js` that returns `{x, y, z}` coordinates
2. Add a button in `index.html` with `data-shape="yourFunctionName"`

That's it. The system auto-generates positions for all 18k particles based on your function.

## Common issues

**"No hands detected"**
- Check your lighting - MediaPipe needs to actually see your hands
- Make sure you're not covering the camera
- Try moving closer/farther from the camera
- Grant camera permissions if the browser blocked it

**Gestures are jittery or unresponsive**
- Improve lighting
- Adjust smoothing factors in config.js
- Make sure your hands are fully visible (not cut off by camera frame)
- If it's laggy, lower particle count

**"Module not found" errors**
- You need a web server. Don't open index.html directly as a file
- Use `python -m http.server` or similar

**Frame rate issues**
- Lower particle count in config.js (try 10000 instead of 18000)
- Disable gradient mode if enabled
- Close other browser tabs

## Performance notes

This renders 18,000 particles at 60fps on a decent GPU. Each frame:
- Updates particle positions (morphing + repulsion physics)
- Optionally calculates per-particle colors for gradients
- Applies bloom post-processing

If you're running this on a potato, reduce particle count. If you have a beast GPU, crank it up to 30k+ and watch it glow.

## License

MIT - do whatever you want with it

## Built with

- [Three.js](https://threejs.org/) - 3D graphics
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) - Google's hand tracking
- Lots of coffee and trial-and-error with shader math
