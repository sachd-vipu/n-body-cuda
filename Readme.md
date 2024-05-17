# Galaxy Evolution: N-body Simulation using Barnes-Hut Approximation Algorithm in CUDA

This project implements an N-body simulation using the Barnes-Hut approximation algorithm in CUDA, designed to simulate the gravitational interactions in galaxy evolution. The implementation utilizes an efficient octree structure and is optimized for performance using CUDA.

## Installation

To install and run this simulation, you need the following:

- **GLM (OpenGL Mathematics):** Install GLM and include its path in your include directory.
- **G++ Compiler:** Ensure you have a modern version of G++ that supports at least C++11.
- **Nvidia Driver and CUDA Toolkit:** Install the latest versions suitable for your hardware.

### Installation Steps

1. Install GLM and include its path in your project.
2. Install the latest versions of the Nvidia driver and CUDA toolkit.
3. Ensure you have a modern G++ compiler installed.

## Usage

To compile and run the simulation:

1. Run the build command:
   ```bash
   make build
   ```
2. Execute the program:
   ```bash
   ./main
   ```

### Configuration

- Modify `GPU_config.hpp` to adjust different configurations and to enable or disable OpenGL animation.
- Define the number of particles as `x*y` where `x` is the grid size and `y` is the block size. Update these values in `kernel_api.cuh` and `barneshut_kernel.cu`.
- In `NBODysimulation.cpp`, use the `setParticlePosition` function to set the initial mass, velocity, and position of the particles.

## Features

- **Octree Representation:** Efficiently handles the computational complexity of the N-body simulation.
- **Array-Based Data Structures:** Enhances performance by optimizing memory access patterns.
- **Flexible Configuration:** Adjust simulation parameters easily through GPU_CONFIG.gpp.

## Credits

- **Paper by Martin Burtscher and Keshav Pingali:** For the theoretical foundation of the implemented algorithm.
- **OpenCL Implementation:** [https://github.com/bneukom/gpu-nbody](#) for reference and comparison.

## Connect

Check out my [LinkedIn post](#) about this project to learn more.
```

