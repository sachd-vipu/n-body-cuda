#include "kernel_api.cuh"
#include "simulation.cu"


void ComputeBoundingBox(float* x, float* y, float *z, float* top, float* bottom, float* right, float* left, float *front, float *back, int* mutex, int p_count)
{
	kernel1_bounding_box_computation<<< params.gridSize, params.blockSize >>>(x, y, z, top, bottom, right, left, front, back, mutex, p_count);
}


void ConstructOctree(float* x, float* y, float *z, float* top, float* bottom, float* right, float* left, float *front, float *back, float* mass, int* count, int *root, int *child,  int* index, int p_count)
{
	kernel2_construct_octree<<< params.gridSize, params.blockSize >>>(x, y, z, top, bottom, right, left, front, back, mass, count, root, child, index, p_count);
}


void ComputeBodyInfo(float* x, float* y, float *z, float *mass, int *index,  int p_count)
{
	kernel3_body_information_octree_node<<<params.gridSize, params.blockSize>>>(x, y, z, mass, index, p_count);
}


void SortBodies(int *count, int *root, int *sorted, int *child, int *index, int p_count)
{
	kernel4_approximation_sorting<<< params.gridSize, params.blockSize >>>(count, root, sorted, child, index, p_count);
}


void CalculateForce(float* x, float *y, float *z,float *vx, float *vy, float *vz, float *ax, float *ay, float *az, float *mass, int *sorted, int *child, float *left, float *right, int p_count)
{
    kernel5_compute_forces_n_bodies<<< params.gridSize, params.blockSize >>>(x, y, z, vx, vy, vz, ax, ay, az, mass, sorted, child, left, right, p_count);
}

void UpdateParticles(float* x, float *y, float *z,  float *vx, float *vy, float *vz, float *ax, float *ay, float *az, int p_count, float dt, float damp)
{
	kernel6_update_velocity_position<<<params.gridSize, params.blockSize >>>(x, y, z, vx, vy, vz, ax, ay, az, p_count, dt, damp);
}

void PopulateCoordinates(float *x, float *y, float *z, float *output, int p_count) 
{
	aux_kernel_copy_3D_coordinate_array<<<params.gridSize, params.blockSize >>>(x, y, z, output ,p_count) ;
}

 void ResetArrays(float *x, float *y, float *z, float *top, float *bottom, float *right, float *left, float *front, float *back, float *mass, int *count, int *root, int* sorted, int *child, int *index, int* mutex, int p_count, int node_count) 
{
	aux_kernel_initialize_device_arrays<<<params.gridSize, params.blockSize >>>(x, y, z,top, bottom, right, left, front, back, mass, count, root,  sorted, child, index,  mutex,  p_count,  node_count);
}

