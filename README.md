# Metaballs 3D Demo

Real-time metaballs demo written in **C++20** and **OpenGL** (SDL2 + GLEW).  
A 3D field is generated on the GPU each frame, then a smooth surface is extracted and rendered with lighting.

## Features

- Animated metaballs
- GPU field generation (**compute shader**)
- On-the-fly surface extraction on GPU (**geometry shader**)
- Smooth shading (normals from field gradient)
- Tonemapping for nicer highlights

## Requirements

- OpenGL **4.3+** (compute shaders)

## Build

**Dependencies**

- CMake 3.x+
- A C++20 compiler
- SDL2
- GLEW

**Commands**

```bash
cmake -S . -B build
cmake --build build -j
```

## Controls

- Mouse: look around
- `W/A/S/D`: move
- `Q/E`: down/up
- `Left Shift`: faster movement
- `Up/Down`: change iso-level
- `Left/Right`: change grid resolution (quality/performance)
- `Esc`: exit

## Notes

- If you change the grid resolution too high, performance will drop quickly (the 3D field is recomputed every frame).
