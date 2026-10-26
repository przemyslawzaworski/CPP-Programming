# CPP-Programming — Examples and Systems Catalog

Various C/C++ examples. DirectX, OpenGL, CUDA, Vulkan, OpenCL.

[Repository](https://github.com/przemyslawzaworski/CPP-Programming)

This catalog describes the supplied local snapshot: **140 files in 105 entries across 17 categories**. Descriptions follow source contents, embedded shaders, data loading and build instructions. Related host/shader/resource files are grouped as one system. Independent executables remain separate, even when they share a folder or filename stem.

## How to use this catalog

- Most examples target Windows. `GLSL/ASM/greenblue.c` uses Linux X11/GLX; Metal shaders require an Apple Metal host. General console utilities may be portable, subject to their includes and file handling.
- Each row describes one example, system, reusable snippet or support bundle. The five `RenderingEngine` programs are alternatives, not translation units to link together.
- Many `.c`/`.cpp` files embed GLSL/HLSL, assembly or compiled shader arrays. The three top-level OpenCL `.cl` files contain host applications and embedded kernels, despite their extension.
- Use the compiler commands in each source as historical starting points. SDK paths, architecture flags, driver extensions and dependency versions may need adjustment. Source analysis does not establish that an example currently builds or runs.
- Links point to the original GitHub paths, so this file can be copied into the repository without changing its links.

## Licensing conventions

**MIT** reflects the repository owner's declaration for their own code and assets. This snapshot has no standalone root LICENSE file. This catalog records the declaration and identified exceptions; it does not replace upstream notices or grant rights to third-party material.

Where a row contains third-party code, the license column separates the author's host/integration code from upstream material. An external dependency's terms apply to that dependency even if the example itself is MIT. **Provenance review** identifies a concrete source/attribution question, not a finding that a license was violated.

For ShaderToy-derived portions, **CC BY-NC-SA 3.0** means **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License**, following the owner's earlier cataloging rule. Individual shader pages should still be checked for explicit author-specific terms before reuse.

## Contents

- [Audio](#audio) — 1 entries
- [Compiler](#compiler) — 1 entries
- [CUDA](#cuda) — 11 entries
- [CPU graphics](#cpu-graphics) — 1 entries
- [OpenGL: setup and drawing](#opengl-setup-and-drawing) — 13 entries
- [OpenGL: compute, bindless and mesh shaders](#opengl-compute-bindless-and-mesh-shaders) — 11 entries
- [OpenGL: effects and rendering](#opengl-effects-and-rendering) — 13 entries
- [OpenGL: textures and scene systems](#opengl-textures-and-scene-systems) — 7 entries
- [OpenGL: audio integration](#opengl-audio-integration) — 1 entries
- [OpenGL: legacy assembly](#opengl-legacy-assembly) — 3 entries
- [DirectX / HLSL](#directx--hlsl) — 17 entries
- [Metal](#metal) — 1 entries
- [OpenCL](#opencl) — 5 entries
- [Vulkan](#vulkan) — 7 entries
- [General utilities](#general-utilities) — 9 entries
- [Bundled third-party support](#bundled-third-party-support) — 3 entries
- [Repository documentation](#repository-documentation) — 1 entries

## Audio

| Files | Short description | License |
| --- | --- | --- |
| [Audio/playsound.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Audio/playsound.c) | Plays an embedded Ogg/Vorbis byte array through the FMOD API. Windows example; FMOD headers and libraries must be supplied separately. | MIT example; separate FMOD terms [L3] |

## Compiler

| Files | Short description | License |
| --- | --- | --- |
| [Compiler/clear.bat](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Compiler/clear.bat)<br>[Compiler/example.pas](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Compiler/example.pas)<br>[Compiler/global.h](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Compiler/global.h)<br>[Compiler/lexer.l](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Compiler/lexer.l)<br>[Compiler/main.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Compiler/main.c)<br>[Compiler/makefile](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Compiler/makefile)<br>[Compiler/parser.y](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Compiler/parser.y)<br>[Compiler/Readme.md](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Compiler/Readme.md) | One Flex/Bison system for lexical and syntax analysis of a Pascal-like language, with shared declarations, driver, sample input, makefile, cleanup script and instructions. Parser actions print diagnostics; this is a front end rather than a complete machine-code compiler. | MIT |

## CUDA

| Files | Short description | License |
| --- | --- | --- |
| [CUDA/abstract.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/abstract.cu) | Animated trigonometric domain-warp pattern computed on CUDA and copied to a Windows GDI bitmap. | MIT |
| [CUDA/apollonian.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/apollonian.cu) | CUDA port of the Apollonian distance-estimator renderer also used by GLSL/ssbo.c and Vulkan/shader.frag; displays the result through GDI. Shared formulas and constants reveal the same shader lineage. | MIT host; CC BY-NC-SA 3.0 shader [L1] |
| [CUDA/bitmap.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/bitmap.cu) | Minimal CUDA pixel-buffer example generating a static procedural image and displaying it through Windows GDI. | MIT |
| [CUDA/blocks.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/blocks.cu) | Animated procedural pixel pattern implemented with inline PTX inside a CUDA kernel; Windows GDI presents the copied image. | MIT |
| [CUDA/eternity.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/eternity.cu) | CUDA/Direct3D 11 interoperability: CUDA writes a shared texture surface and Direct3D draws it to the window. | MIT |
| [CUDA/multipass.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/multipass.cu) | Persistent CUDA image buffer: reads the previous pixel value and accumulates a moving circular mark across frames; GDI display. | MIT |
| [CUDA/sphere.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/sphere.cu) | Offline sphere raymarcher with lighting; CUDA generates an image saved as sphere.ppm. Requires the external CUDA sample helper_math.h header. | MIT |
| [CUDA/surface.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/surface.cu) | CUDA/OpenGL image interoperability using a registered texture and CUDA surface writes, followed by OpenGL display. | MIT |
| [CUDA/TensorCore.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/TensorCore.cu) | Small WMMA matrix multiplication using half-precision inputs and float accumulation on Tensor Cores; the supplied build targets sm_75. | MIT |
| [CUDA/tetrahedron.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/tetrahedron.cu) | Raymarches a rotating tetrahedron built from plane distance fields, with directional-derivative lighting; CUDA/OpenGL buffer interoperability and FreeGLUT. References external tutorials and CUDA sample helpers. | MIT own code; upstream provenance review [L2] |
| [CUDA/uv.cu](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/uv.cu) | CUDA/OpenGL vertex-buffer interoperability: generates a pixel-point grid with coordinate-based colors and draws it with OpenGL. | MIT |

## CPU graphics

| Files | Short description | License |
| --- | --- | --- |
| [GFX/tunnel.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GFX/tunnel.c) | Multithreaded software tunnel renderer with a Windows GDI framebuffer; source comments discuss screen tearing. | MIT |

## OpenGL: setup and drawing

| Files | Short description | License |
| --- | --- | --- |
| [GLSL/blank.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/blank.c) | Minimal OpenGL 4.5 program-pipeline example displaying an animated cosine color field. | MIT |
| [GLSL/buffer.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/buffer.c) | Small OpenGL compute/SSBO example: writes a vec4 on the GPU and reads the buffer back on the CPU. | MIT |
| [GLSL/context.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/context.c) | Windows OpenGL context and pixel-format setup example with a basic triangle. | MIT |
| [GLSL/GLSL.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/GLSL.c) | Compact Windows OpenGL/GLSL example rendering a fullscreen coordinate-color gradient. | MIT |
| [GLSL/GLSL.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/GLSL.cpp) | Separate GLEW-based GLSL 1.30 example with a coordinate/time color shader; an alternative executable to GLSL.c. | MIT |
| [GLSL/GLSL4.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/GLSL4.cpp) | GLEW-based OpenGL 4.5 separable-program pipeline demo with an animated coordinate gradient. | MIT |
| [GLSL/quad.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/quad.c) | Indexed quad drawing using vertex/index buffers and a vertex array object. | MIT |
| [GLSL/triangle.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/triangle.c) | Basic shader-driven triangle with per-vertex colors and vertex-buffer/vertex-array setup. | MIT |
| [GLSL/vs_quad.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/vs_quad.c) | Generates a fullscreen quad from gl_VertexID without uploading a conventional vertex buffer. | MIT |
| [GLSL/geometry.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/geometry.c) | Geometry shader expands four input points into differently colored eight-segment closed line outlines. | MIT |
| [GLSL/fixed.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/fixed.c) | CPU-generated point positions/colors displayed using OpenGL fixed-function drawing. | MIT |
| [GLSL/menu.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/menu.c) | Win32 rendering-settings dialog followed by an OpenGL shader window. | MIT |
| [GLSL/mvp.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/mvp.c) | Manual model/view/projection matrix and camera-control example using keyboard and mouse input. | MIT |

## OpenGL: compute, bindless and mesh shaders

| Files | Short description | License |
| --- | --- | --- |
| [GLSL/ComputeShader.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/ComputeShader.c) | Compute shader generates a checkerboard texture and displays it through OpenGL. | MIT |
| [GLSL/ComputeShader.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/ComputeShader.cpp) | Independent GLEW demo: computes a checkerboard texture, then samples it on a raymarched rotating cube using triplanar mapping. | MIT |
| [GLSL/flames.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/flames.c) | Compute-shader adaptation of a ShaderToy fire effect; writes a texture for fullscreen display. | MIT host; CC BY-NC-SA 3.0 shader [L1] |
| [GLSL/ssbo.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/ssbo.c) | Compute shader renders an Apollonian fractal into an SSBO; CPU mapping and Windows GDI display the packed pixels. Same shader core as Vulkan/shader.frag. | MIT host; CC BY-NC-SA 3.0 shader [L1] |
| [GLSL/stain.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/stain.c) | Compute shader generates an animated stain-like color field using repeated trigonometric transformations; writes an image2D each frame. | MIT |
| [GLSL/bindless.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/bindless.c) | NVIDIA bindless-texture demo drawing many texture-backed elements using GPU texture handles. Requires an external heightmap texture collection linked in the source. | MIT |
| [GLSL/fireball.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/fireball.c) | NVIDIA task/mesh-shader example generating a fireball-like point effect with shader-side sphere intersection calculations. | MIT |
| [GLSL/grid.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/grid.c) | NVIDIA task/mesh shaders generate and draw a procedural point grid. | MIT |
| [GLSL/ms_triangle.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/ms_triangle.c) | Minimal NVIDIA mesh-shader triangle example. | MIT |
| [GLSL/nvstream.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/nvstream.c) | Task/mesh-shader adaptation of the VertexShaderArt stream effect; generates animated colored line geometry. | MIT own code; upstream provenance review [L2] |
| [GLSL/stream.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/stream.c) | Conventional vertex-shader version of the animated colored line stream, with geometry generated from gl_VertexID. | MIT own code; upstream provenance review [L2] |

## OpenGL: effects and rendering

| Files | Short description | License |
| --- | --- | --- |
| [GLSL/BlueTunnel.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/BlueTunnel.c) | Fullscreen procedural blue tunnel fragment shader; the quoted gl.h include needs a suitable header search path. | MIT |
| [GLSL/mandelbulb.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/mandelbulb.c) | Fullscreen raymarching of a Mandelbulb fractal with distance-estimator shading. | MIT |
| [GLSL/smoke.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/smoke.c) | Volumetric smoke rendered by accumulating density/color samples along a ray; uses a CPU-generated noise texture. Shader provenance deserves review because its density/noise structure resembles established cloud examples. | MIT own code; upstream provenance review [L2] |
| [GLSL/Terrain.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Terrain.c) | Procedural terrain raymarcher using layered noise, estimated surface normals and lighting. | MIT |
| [GLSL/PaintWithShadows.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/PaintWithShadows.c) | Interactive multipass painting effect with mouse input and shadow/height-based shading of the accumulated image. | MIT |
| [GLSL/PostProcessing.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/PostProcessing.c) | Renders a procedural fractal volume to a framebuffer texture, then converts it to grayscale in a second fullscreen pass. | MIT |
| [GLSL/strangefluid.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/strangefluid.c) | ShaderToy-derived fluid-like feedback effect using multiple render passes and previous-frame textures. | MIT host; CC BY-NC-SA 3.0 shader [L1] |
| [GLSL/pathtracing1.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/pathtracing1.c) | Progressive path tracing with framebuffer accumulation, randomized hemisphere sampling and multiple diffuse bounces; repeated cube/sphere CSG scene and ground plane. | MIT own code; upstream provenance review [L2] |
| [GLSL/pathtracing2.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/pathtracing2.c) | Independent path-tracing variant of pathtracing1.c with a hollow box/interior scene, altered camera and ten-bounce integration. | MIT own code; upstream provenance review [L2] |
| [GLSL/Demo.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Demo.cpp) | GLEW demoscene-style program combining procedural fullscreen visuals, generated noise and CPU-synthesized audio played through WinMM. | MIT |
| [GLSL/multicore.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/multicore.cpp) | CPU image generation split across worker threads, with OpenGL used to display the resulting texture. | MIT |
| [GLSL/GenerateTexture.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/GenerateTexture.cpp) | Generates a noise texture on the CPU and displays it with animated shader modulation using GLEW. | MIT |
| [GLSL/VolumeParticleSystem.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/VolumeParticleSystem.c) | Standalone 3D scene with animated cube particles generated from vertex IDs, atmospheric sky, ground, camera controls and a framebuffer vignette. The sky shader contains Unity-style atmospheric-scattering code; review upstream attribution. | MIT own code; upstream provenance review [L2] |

## OpenGL: textures and scene systems

| Files | Short description | License |
| --- | --- | --- |
| [GLSL/LoadTexture.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/LoadTexture.c)<br>[GLSL/clouds.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/clouds.ppm)<br>[GLSL/plaster.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/plaster.ppm) | Binary PPM texture loader and two supplied texture images; blends/samples textures in an OpenGL shader. plaster.ppm is also used by the basic RenderingEngine demos. | MIT |
| [GLSL/Skybox/skybox.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Skybox/skybox.c)<br>[GLSL/Skybox/textures/0.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Skybox/textures/0.ppm)<br>[GLSL/Skybox/textures/1.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Skybox/textures/1.ppm)<br>[GLSL/Skybox/textures/2.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Skybox/textures/2.ppm)<br>[GLSL/Skybox/textures/3.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Skybox/textures/3.ppm)<br>[GLSL/Skybox/textures/4.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Skybox/textures/4.ppm)<br>[GLSL/Skybox/textures/5.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Skybox/textures/5.ppm)<br>[GLSL/Skybox/textures/plaster.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Skybox/textures/plaster.ppm) | One skybox scene: six supplied cubemap faces (0.ppm–5.ppm), ground plaster texture, skybox shaders and camera/scene code. | MIT |
| [GLSL/RenderingEngine/demo.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/RenderingEngine/demo.c)<br>[GLSL/RenderingEngine/compile.cmd](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/RenderingEngine/compile.cmd) | Basic textured geometry/camera example and its compile helper. Loads plaster.ppm by working-directory-relative path; supplied texture is one folder above. | MIT |
| [GLSL/RenderingEngine/demo.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/RenderingEngine/demo.cpp) | Independent C++/WinMain version of the basic textured geometry and camera example; also expects plaster.ppm in the working directory. | MIT |
| [GLSL/RenderingEngine/demo1.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/RenderingEngine/demo1.c) | Scene framework with first-person camera, simple collision tests, cubemap sky, textured geometry and looping WAV ambience. Several textures and wind.wav are not supplied. | MIT |
| [GLSL/RenderingEngine/demo2.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/RenderingEngine/demo2.c) | Larger scene framework with OBJ loading, textured objects, collision tests, skybox, particle surface, post-processing and WAV ambience. Required geometry, textures and audio folders are absent. | MIT |
| [GLSL/RenderingEngine/demo3.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/RenderingEngine/demo3.c) | Forest-scene variant adding depth/shadow-map passes, alpha-masked trees and grass. Required OBJ models, scene textures and WAV audio are absent. | MIT |

## OpenGL: audio integration

| Files | Short description | License |
| --- | --- | --- |
| [GLSL/Audio.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/Audio.c) | Procedural fullscreen visuals with a grayscale post-process and 4klang-generated music played through WinMM. Requires an exported 4klang.obj song object that is not included; generated 4klang.h supplies replacement constants; the DLL below is the authoring plugin, not the linked song object. | MIT own code; upstream 4klang terms [L4] |

## OpenGL: legacy assembly

| Files | Short description | License |
| --- | --- | --- |
| [GLSL/ASM/cellular.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/ASM/cellular.c) | Windows ARB fragment-program assembly example producing an animated cellular pattern. | MIT |
| [GLSL/ASM/greenblue.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/ASM/greenblue.c) | Linux X11/GLX ARB fragment-program assembly example producing a green/blue color effect. | MIT |
| [GLSL/ASM/redblue.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/ASM/redblue.c) | Windows ARB vertex/fragment-program assembly example producing a red/blue gradient; references an external GPU programming guide. | MIT |

## DirectX / HLSL

| Files | Short description | License |
| --- | --- | --- |
| [HLSL/Asm.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/Asm.cpp) | Direct3D 9 assembles vs_2_0/ps_2_0 text at runtime and renders a solid blue fullscreen quad; requires legacy D3DX9. | MIT |
| [HLSL/Audio.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/Audio.cpp) | Direct3D 9 shader visuals with synthesized PCM audio played through WinMM; includes its own synthesis code rather than including audio.h. | MIT |
| [HLSL/audio.h](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/audio.h) | Reusable bass-synthesis function snippet. No entry point; Audio.cpp does not include this header. | MIT |
| [HLSL/checkerboard.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/checkerboard.cpp) | Direct3D 11 sample generating a checkerboard texture on the CPU and sampling it on a fullscreen quad. | MIT |
| [HLSL/Circle.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/Circle.cpp) | Direct3D 9 pixel-shader assembly demo rendering a circle. | MIT |
| [HLSL/firesmoke.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/firesmoke.c) | Direct3D 12 fire/smoke visual demo with embedded compiled shader bytecode; original HLSL source files are not bundled. | MIT |
| [HLSL/Helix.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/Helix.cpp) | Direct3D 9 pixel-shader raymarching of a helix-shaped procedural object. | MIT |
| [HLSL/hexmap.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/hexmap.c) | Direct3D 9 assembly-shader procedural hexagonal pattern. | MIT |
| [HLSL/mosaic.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/mosaic.c) | Direct3D 12 mosaic-pattern demo with embedded compiled shaders rather than separate HLSL sources. | MIT |
| [HLSL/neon.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/neon.c) | Direct3D 11 compute-driven neon-style image demo using embedded compiled shader bytecode. | MIT |
| [HLSL/notepad.txt](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/notepad.txt) | Direct3D 12 video-memory usage/budget query snippet; supporting reference text rather than a standalone application. | MIT |
| [HLSL/particles.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/particles.c) | Direct3D 11 fullscreen pixel effect evaluating moving circular particles and accumulating samples to create light streaks; not a vertex-buffer particle simulation. | MIT |
| [HLSL/Pattern.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/Pattern.cpp) | Direct3D 9 procedural fullscreen pattern shader. | MIT |
| [HLSL/SM3_HelloWorld.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/SM3_HelloWorld.cpp) | Minimal Direct3D 9 Shader Model 3 program with embedded vertex/pixel HLSL and a coordinate/time-based color output. | MIT |
| [HLSL/tetrahedron.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/tetrahedron.c) | Direct3D 11 compute shader raymarches a tetrahedron from a signed-distance field. | MIT |
| [HLSL/tga.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/tga.cpp) | Direct3D 11 32-bit TGA texture loading, mip generation and runtime texture switching/unloading. Expected bricks.tga/plasma.tga assets are absent. | MIT |
| [HLSL/texture.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/texture.cpp)<br>[HLSL/plasma.ppm](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/HLSL/plasma.ppm) | Direct3D 12 textured 3D rendering and camera example, with a binary PPM texture loader and the supplied plasma image. | MIT |

## Metal

| Files | Short description | License |
| --- | --- | --- |
| [MetalAPI/Line.metal](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/MetalAPI/Line.metal) | Metal vertex/fragment shader pair drawing an antialiased 2D line segment on a fullscreen primitive. Contains generated-style array helpers; no Metal host application is supplied. | MIT |

## OpenCL

| Files | Short description | License |
| --- | --- | --- |
| [OpenCL/dawn.cl](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/OpenCL/dawn.cl) | Windows OpenCL host application with an embedded kernel/PTX-assisted math producing an animated dawn-like image; GDI display. | MIT |
| [OpenCL/plasma.cl](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/OpenCL/plasma.cl) | Windows OpenCL host application compiling an embedded plasma kernel and copying pixels back for GDI display. | MIT |
| [OpenCL/waves.cl](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/OpenCL/waves.cl) | Windows OpenCL host application with an embedded animated waves kernel and GDI output. | MIT |
| [OpenCL/Tools/clcc.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/OpenCL/Tools/clcc.c) | Command-line OpenCL source compilation checker that creates a context/program and reports compiler build logs. | MIT |
| [OpenCL/Tools/clptx.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/OpenCL/Tools/clptx.c) | Compiles OpenCL input and exports the device program binary to output.ptx; PTX interpretation assumes an NVIDIA OpenCL implementation. | MIT |

## Vulkan

| Files | Short description | License |
| --- | --- | --- |
| [Vulkan/apollonian.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/apollonian.c)<br>[Vulkan/shader.frag](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/shader.frag) | One Win32 Vulkan Apollonian-fractal system: host code embeds compiled SPIR-V matching the supplied fragment shader. Uses push-constant time; shader.frag contains the ShaderToy source link. | MIT host; CC BY-NC-SA 3.0 shader [L1] |
| [Vulkan/chess.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/chess.cpp) | GLFW/Vulkan checker-pattern example with embedded vertex/fragment SPIR-V; no separate shader files are needed for those arrays. | MIT |
| [Vulkan/monster.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/monster.c) | Win32 Vulkan procedural animated fractal/pattern demo with embedded SPIR-V; a separate executable from apollonian.c. | MIT |
| [Vulkan/pattern.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/pattern.c) | Win32 Vulkan fullscreen trigonometric pattern using embedded SPIR-V. The generic shader.frag compile comment does not make the current Apollonian shader its matching source. | MIT |
| [Vulkan/vulkan_hello_world.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/vulkan_hello_world.cpp) | GLFW/Vulkan triangle setup based on vulkan-tutorial.com. Loads vert.spv and frag.spv, which are not included in this snapshot. | MIT own changes; upstream tutorial code CC0 1.0 [L5] |
| [Vulkan/RTX/Triangle/build-shaders.bat](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/RTX/Triangle/build-shaders.bat)<br>[Vulkan/RTX/Triangle/hit.rchit](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/RTX/Triangle/hit.rchit)<br>[Vulkan/RTX/Triangle/hit.spv](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/RTX/Triangle/hit.spv)<br>[Vulkan/RTX/Triangle/miss.rmiss](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/RTX/Triangle/miss.rmiss)<br>[Vulkan/RTX/Triangle/miss.spv](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/RTX/Triangle/miss.spv)<br>[Vulkan/RTX/Triangle/raygen.rgen](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/RTX/Triangle/raygen.rgen)<br>[Vulkan/RTX/Triangle/raygen.spv](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/RTX/Triangle/raygen.spv)<br>[Vulkan/RTX/Triangle/triangle.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/RTX/Triangle/triangle.c) | One legacy NVIDIA ray-traced triangle system: Win32 Vulkan host, ray-generation/closest-hit/miss GLSL stages, compiled SPIR-V and shader-build script. Uses NV ray tracing rather than the newer KHR ray-tracing API. | MIT |
| [Vulkan/VulkanComputeShader/demo.comp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/VulkanComputeShader/demo.comp)<br>[Vulkan/VulkanComputeShader/demo.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/VulkanComputeShader/demo.cpp)<br>[Vulkan/VulkanComputeShader/demo.spv](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/VulkanComputeShader/demo.spv) | One Vulkan compute system: host, editable GLSL compute shader and compiled SPIR-V. Samples an input checkerboard into an animated perspective floor, reads the output image back and displays it through Windows GDI. | MIT |

## General utilities

| Files | Short description | License |
| --- | --- | --- |
| [Tools/BaseConverter.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Tools/BaseConverter.c) | Console integer numeral-base conversion example in C. | MIT |
| [Tools/generate_html.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Tools/generate_html.cpp) | Writes a minimal Hello World index.html file and opens it through the Windows file association. | MIT |
| [Tools/ImageToArray.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Tools/ImageToArray.c) | Converts binary PPM image data into a C unsigned-char array/header for embedding images. | MIT |
| [Tools/list.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Tools/list.c) | Small dynamically allocated float-list implementation demonstrating append, removal and resizing. | MIT |
| [Tools/MapToArray.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Tools/MapToArray.c) | Converts binary float map data into an embeddable C float array. | MIT |
| [Tools/number_system_converter.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Tools/number_system_converter.cpp) | Independent C++ console number-base conversion example; an alternative to BaseConverter.c. | MIT |
| [Tools/ppm.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Tools/ppm.c) | Command-line binary PPM viewer using a Windows GDI window. | MIT |
| [Tools/registry.cpp](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Tools/registry.cpp) | Windows Registry read/write example. | MIT |
| [Tools/TriangleArea.c](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Tools/TriangleArea.c) | Triangle-area calculation together with a mesh-data structure and binary serialization example. | MIT |

## Bundled third-party support

| Files | Short description | License |
| --- | --- | --- |
| [GLSL/GL/inc/eglew.h](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/GL/inc/eglew.h)<br>[GLSL/GL/inc/glew.h](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/GL/inc/glew.h)<br>[GLSL/GL/inc/glxew.h](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/GL/inc/glxew.h)<br>[GLSL/GL/inc/wglew.h](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/GL/inc/wglew.h)<br>[GLSL/GL/lib/glew32.dll](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/GL/lib/glew32.dll) | Bundled GLEW/EGL/GLX/WGL headers and Windows glew32.dll. Shared support files used by selected examples, not a separate effect. Preserve all GLEW, Mesa and Khronos notices in the headers. | Upstream BSD-style + MIT notices [L6] |
| [GLSL/glext.h](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/glext.h) | Khronos OpenGL extension declarations; shared API header with its own permissive copyright/permission notice. | Upstream MIT-style Khronos notice [L6] |
| [GLSL/4klang.dll](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/GLSL/4klang.dll) | Bundled 4klang synthesizer VST plugin used to author/export music for the GLSL/Audio.c workflow. Current upstream is MIT; exact bundled binary version is not established. | Upstream MIT; binary version unverified [L4] |

## Repository documentation

| Files | Short description | License |
| --- | --- | --- |
| [README.md](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/README.md)<br>[CUDA/readme.md](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/CUDA/readme.md)<br>[OpenCL/README.md](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/OpenCL/README.md)<br>[Vulkan/readme.md](https://github.com/przemyslawzaworski/CPP-Programming/blob/master/Vulkan/readme.md) | Repository overview and API-folder notes/build instructions; these are documentation rather than additional runnable examples. | MIT |

## License and provenance findings

| Reference | Affected material | Finding and follow-up |
| --- | --- | --- |
| L1 | `GLSL/flames.c`; `GLSL/strangefluid.c`; Apollonian family | Direct ShaderToy references: [fire](https://www.shadertoy.com/view/4ttGWM), [fluid](https://www.shadertoy.com/view/XdcGW2), [Apollonian](https://www.shadertoy.com/view/4d2BW1). `Vulkan/shader.frag` identifies the Apollonian source. `CUDA/apollonian.cu` and `GLSL/ssbo.c` reproduce the same seven-iteration folded/inverted distance estimator and the same ray setup, step count and brightness expression without repeating the link. `Vulkan/apollonian.c` embeds the matching shader as SPIR-V. These shader portions should not be presented as unqualified MIT. Preserve upstream author credit and the original link in each port. |
| L2 | `GLSL/stream.c`, `GLSL/nvstream.c` | Both cite [the same VertexShaderArt artwork](https://www.vertexshaderart.com/art/ZSksx2deRsDocFDKT). Its individual authorship/license could not be verified from the retrieved page. Do not infer an artwork's license from the website engine's license; confirm the creator and permission for the adapted shader. |
| L2 | `GLSL/pathtracing1.c`, `GLSL/pathtracing2.c` | Both explicitly reference [Inigo Quilez's simple path-tracing article](https://iquilezles.org/articles/simplepathtracing/). The source does not include an upstream license notice. Confirm the permission applicable to reused shader code; MIT applies to the owner's own additions. A reference alone does not prove that every line is copied. |
| L2 | `CUDA/tetrahedron.cu` | References the [GPU path-tracing tutorial](https://github.com/straaljager/GPU-path-tracing-tutorial-2), an Inigo Quilez lighting article and NVIDIA's simpleGL sample. The file credits the repository owner. Check attribution for any reused interoperability/sample code; separately preserve notices for external helper headers. |
| L2 | `GLSL/smoke.c`, `GLSL/VolumeParticleSystem.c` | Additional provenance leads from code inspection: smoke uses a recognizable layered-noise/cloud-density structure; the particle scene includes an atmospheric sky with Unity-style identifiers such as `_SkyTint`, `_AtmosphereThickness`, `kHDSundiskIntensityFactor` and matching scattering calculations. These are review leads rather than confirmed license assignments. Confirm the original versions and retain their notices before describing the entire files as exclusively MIT. |
| L3 | `Audio/playsound.c` | [FMOD has separate licensing](https://www.fmod.com/licensing), including qualifying free use and paid licensing. The example's MIT status does not cover FMOD. SDK files are not included. The origin/permission of the embedded Ogg recording is not documented in this source and should be confirmed separately from the playback code. |
| L4 | `GLSL/4klang.dll`, generated music workflow | [Current 4klang upstream LICENSE](https://github.com/gopher-atz/4klang/blob/master/LICENSE) is MIT. The precise bundled DLL version cannot be established from this source review. Preserve its applicable original notice; do not substitute the example author's copyright. Exported music/song objects may have their own authorship. The Audio.c instructions mention Renoise as a VST host; this does not establish that a paid host is required. |
| L5 | `Vulkan/vulkan_hello_world.cpp` | [Vulkan Tutorial](https://vulkan-tutorial.com/Introduction) distinguishes CC0 1.0 source code from CC BY-SA 4.0 tutorial prose. The example references that tutorial; preserve the distinction rather than labeling third-party prose MIT. |
| L6 | GLEW headers/DLL and `GLSL/glext.h` | Header contents contain their own GLEW/Mesa/Khronos permissive notices, including BSD-style and MIT-style text. Preserve these notices. [GLEW licensing information](https://glew.sourceforge.net/credits.html) provides the upstream reference; the bundled header notices are the direct evidence for this snapshot. |

## Build and resource notes

| Area | Practical requirement or limitation |
| --- | --- |
| General Windows examples | Windows headers/libraries and the compiler flavor expected by each command. Some sources use old MinGW/MSVC, legacy OpenGL contexts or DirectX SDK paths. |
| GLEW examples | Supplied headers and DLL are under `GLSL/GL/`. Commands using `-lglew32` still require a matching import library; none is included. Header/library paths and DLL architecture must agree. |
| NVIDIA-specific OpenGL | `bindless.c` uses NVIDIA bindless texture support; `fireball.c`, `grid.c`, `ms_triangle.c` and `nvstream.c` use `GL_NV_mesh_shader`. A generic OpenGL version check alone is insufficient. |
| CUDA | NVIDIA CUDA toolkit/GPU; several commands target historical `sm_30`. Helper headers, FreeGLUT and OpenGL setup are external for selected examples. `TensorCore.cu` has a different WMMA-capable hardware target. |
| DirectX | Direct3D 9 examples using D3DX need the legacy D3DX components. Direct3D 11/12 examples require their corresponding Windows SDK/runtime support. Several shader blobs have no editable matching HLSL source in this snapshot. |
| OpenCL | OpenCL headers, loader and an implementation/device. The `.cl` host examples must be compiled as host source as their comments describe; they are not input files for clcc without extracting the embedded kernel. |
| Vulkan | Vulkan SDK/runtime. GLFW is additionally required by `chess.cpp` and `vulkan_hello_world.cpp`. RTX/Triangle specifically uses the older NV ray-tracing extension and its matching shader/toolchain support. |
| Vulkan shader files | `shader.frag` matches Apollonian, not every program mentioning that generic filename in a compile comment. Other Vulkan programs embed their own SPIR-V. `vulkan_hello_world.cpp` expects missing `vert.spv`/`frag.spv`; RTX and compute subfolders include their shader sources and binaries. |
| External assets | `bindless.c` links an external texture archive. RenderingEngine `demo1.c`–`demo3.c` reference unbundled textures/models/WAVs; `tga.cpp` expects absent TGA images. Asset availability and licenses are separate from source-code licensing. |
| Shared PPM images | Basic RenderingEngine demos expect `plaster.ppm` in their working directory, while the supplied file is in `GLSL/`. The Skybox folder is a complete grouped resource example, but does not supply every asset referenced by other scene demos. |
| 4klang audio | `GLSL/Audio.c` requires generated song exports. The bundled VST DLL does not replace missing `4klang.obj`; the generated `4klang.h` is used to update constants already present in the C source. |
| Compiler | Requires Flex/Bison and a compatible build environment. The makefile mixes generated `.c` source names with `.cpp` object-substitution/rule conventions; inspect the rules before relying on an automated build. |
| Metal | Only the shader is supplied; integrate it into a Metal application with the expected entry points and uniforms. |

## Coverage and scope

Every file in the supplied local snapshot is linked in exactly one catalog row, including headers, binary textures, DLLs, compiled shaders, build scripts and documentation. Shared dependencies can be mentioned by several descriptions without creating duplicate catalog entries. File grouping follows code dependencies and application entry points rather than filename similarity alone.

This is a source/documentation inventory. Programs, DLLs and build/cleanup scripts were not executed, and GPU output was not visually validated. License exceptions above record observed notices, source references and explicitly identified review questions.
