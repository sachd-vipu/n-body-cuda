#ifndef GPU_CONFIG
#define GPU_CONFIG

extern int COMPUTE_UNITS = 20;
constexpr int WARP_SIZE = 32;
constexpr float PI = 3.14159266;
constexpr float DAMP = 1.0;
constexpr float GRAVITY = 1.0;
constexpr int ITERATIONS = 80;
constexpr float TIMESTEP = 0.001;
constexpr bool PLOT_OPENGL = false;
constexpr bool GENERATE_PARTICLEDATA = true;
constexpr int GRID_SIZE = 1024;
constexpr int BLOCK_SIZE = 1024;
#endif 

