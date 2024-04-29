__global__ void aux_kernel_initialize_device_arrays(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *root, int *sorted, int *child, int *index, float *left, float *right, float *bottom, float *top, float* front, float* back, int p_count, int node_count);
__global__ void kernel1_bounding_box_computation(int *mutex, float *x, float *y, float *z, float *left, float *right, float *bottom, float *top, float*front, float* back, int p_count);
__global__ void kernel2_construct_octree(float *x, float *y, float *z, float *mass, int *count, int *root, int *child, int *index, float *left, float *right, float *bottom, float *top, float*front, float *back, int p_count);
__global__ void kernel3_body_information_octree_node(float *x, float *y, float *z, float *mass, int *index, int p_count);
__global__ void kernel4_approximation_sorting(int *count, int *root, int *sorted, int *child, int *index, int p_count);
__global__ void kernel5_compute_forces_n_bodies(float* x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay, float *az, float *mass, int *sorted, int *child, float *left, float *right, int p_count, float gravity);
__global__ void kernel6_update_velocity_position(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay, float *az, int p_count, float dt, float damp);
__global__ void aux_kernel_copy_3D_coordinate_array(float* x, float* y, float *z, float* out, int p_count);
__global__ void aux_kernel_plot_3D_points(float *out, float *x, float *y, float *z, int p_count)
