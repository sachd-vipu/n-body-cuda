# Galaxy Evolution: N-body Simulation using Barnes-Hut Approximation Algorithm in CUDA

This project implements an N-body simulation using the Barnes-Hut approximation algorithm in CUDA, designed to simulate the gravitational interactions in galaxy evolution. The implementation utilizes an efficient octree structure and is optimized for performance using CUDA.

## Installation

To install and run this simulation, you need the following:

- **GLM (OpenGL Mathematics):** Install GLM and include its path in your include directory.
- **G++ Compiler:** Ensure you have a modern version of G++ that supports at least C++11.
- **Nvidia Driver and CUDA Toolkit:** Install the latest versions suitable for your hardware.

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


### **Project Overview:**
- **Optimized Algorithm**: Modeled the gravitational interactions between particles representing stars as phycial bodies to simulate aspects of galaxy formation and evolution, implementing Barnes-Hut algorithm for approximating gravitational force calucation between particles, aiming for O(n log n) complexity compare to O(n^2) for all pairs.
- **Performance**:  Utilized CUDA to speed up computation of single precision floating point numbers, with simulation for one timestep taking ~3.6 seconds for 1 million particles on 966 Mhz NVIDIA RTX 3050 6 GB Laptop GPU with 2048 cores.


## Features

- **Octree Representation:** Efficiently handles the computational complexity of the N-body simulation.
- **Array-Based Data Structures:** Enhances performance by optimizing memory access patterns.
- **Flexible Configuration:** Adjust simulation parameters easily through GPU_CONFIG.hpp.


### Technical Insights:
- **GPU Utilization**: Used NVIDIA RTX 3050 to handle the massive computations.
- **Visualization**: Leveraged Zindaiji (named after a place near NAOJ) to visualize over 12 GB of particle data for a single simulation run for colliding two galaxies.
- **Learning Experience**: A great refresher on polar coordinates, 3D geometry, and the gravitational physics. 

🔗 Learn more about Zindaiji: [Zindaiji](https://4d2u.nao.ac.jp/src_4d2u_dome/src/Zindaiji/ZINDAIJI3_4d/Zindaiji3Top_E.html)


## Credits

- **Paper by Martin Burtscher and Keshav Pingali:** THE approach was outlined by the work of Martin Burtscher and Keshav Pingali, detailed in their paper on CUDA implementations of the Barnes-Hut algorithm ([Read the paper](https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf)).
- **OpenCL Implementation:** Practical Insights were also drawrn from  [https://github.com/bneukom/gpu-nbody](https://github.com/bneukom/gpu-nbody).

## Connect

Check out my [LinkedIn post](https://www.linkedin.com/posts/sachdvipu_cuda-highperformancecomputing-computationalphysics-activity-7197543932879155200-G4nY?utm_source=share&utm_medium=member_desktop) about this project to learn more.
