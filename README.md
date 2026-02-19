<div align="center">

# SpatialGraph

## AR Drawing Engine


### A Shader Driven Holographic Augmented Reality Rendering System

Built with precision graphics engineering, SpatialGraph bridges real-time computer vision with GPU accelerated rendering to create a spatial computing experience directly from your webcam.

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[![OpenGL](https://img.shields.io/badge/OpenGL-3.3_Core-5586A4?style=for-the-badge\&logo=opengl\&logoColor=white)](https://opengl.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tracking-00A67E?style=for-the-badge\&logo=google\&logoColor=white)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)](https://opencv.org/)

<br/>

---

## Overview

SpatialGraph is not a beginner OpenGL script.

It is a modular, multi pass rendering engine that fuses AI-based hand tracking with modern OpenGL (Core Profile 3.3) to produce volumetric, glowing holographic strokes in real-world space.

Draw dynamic energy ribbons directly into your environment using only your webcam and hand gestures.

<br/>

---

## Cinematic Capabilities


### Volumetric Ribbon Generation

Dynamic triangle-strip generation using velocity-based tangent vectors. Stroke thickness adapts naturally to hand movement speed, creating fluid energy trails.

### Cinematic Bloom Pipeline

Multi pass HDR rendering pipeline with ping-pong Gaussian blur across Framebuffer Objects (FBOs). Glow is composited additively for a high-end holographic aesthetic.

### Spatial EMA Filtering

Implements Exponential Moving Average smoothing to convert raw MediaPipe landmark data into stable, cinematic 3D motion.

### Deferred Real-World Compositing

Camera feed is rendered independently from holographic geometry. Post-processing affects only digital assets, preserving the realism of the physical world.

<br/>

---

## Engine Architecture

SpatialGraph follows structured graphics engine paradigms and strictly avoids deprecated immediate-mode rendering.

**Input Layer**
OpenCV + MediaPipe real-time spatial tracking

**Mathematics Layer**
PyGLM for projection matrices, camera transforms, and vector math

**Memory Layer**
Pre-allocated VBO streaming to eliminate CPU–GPU stalls

**Rendering Layer**
ModernGL with GLSL 330 Core shaders

**Post-Processing Layer**
HDR Framebuffers, blur passes, and additive compositing

<br/>

---

## Installation & Setup

Run the engine locally in a few steps:

```bash
# Clone the repository
git clone https://github.com/YourUsername/SpatialGraph.git
cd SpatialGraph

# Create virtual environment
python3 -m venv arenv
source arenv/bin/activate  # Windows: arenv\Scripts\activate

# Install dependencies
pip install moderngl pygame opencv-python mediapipe numpy PyGLM

# Launch the engine
python main.py
```

<br/>

---

## Performance Targets

* 60 FPS real-time rendering
* GPU accelerated shader pipeline
* Minimal CPU–GPU synchronization overhead
* Stable OpenGL 3.3 Core Profile context

<br/>


---

## Author

### Built by Sarthak Bhopale

Crafted as a graphics engineering exploration into real-time AR rendering systems.

</div>
