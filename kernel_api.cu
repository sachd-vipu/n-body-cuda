#include "kernel_api.cuh"
#include "barneshut_kernel.cuh"


void ComputeBoundingBox(int *mutex, float *x, float *y, float *z, float *left, float *right, float *bottom, float *top, float* front, float *back, int p_count)
{
	kernel1_bounding_box_computation<<<params.gridSize, params.blockSize>>>(mutex, x, y, z, left, right, bottom, top, front, back, p_count);
}
 
void ConstructOctree(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index, float *left, float *right, float *bottom, float *top, float* front, float *back, int p_count)
{
	kernel2_construct_octree<<<params.gridSize, params.blockSize>>>(x, y, z, mass, count, start, child, index, left, right, bottom, top, front, back, p_count);
}
void ComputeBodyInfo(float *x, float *y, float *z, float *mass, int *index, int p_count)
{
	kernel3_body_information_octree_node<<<params.gridSize, params.blockSize>>>(x, y, z, mass, index, p_count);
}

void SortBodies(int *count, int *start, int *sorted, int *child, int *index, int p_count)
{
	kernel4_approximation_sorting<<<params.gridSize, params.blockSize>>>(count, start, sorted, child, index, p_count);

}

void CalculateForce(float* x, float *y, float * z, float *vx, float *vy, float *vz, float *ax, float *ay, float *az, float *mass, int *sorted, int *child, float *left, float *right, int p_count, float gravity)
{
	kernel5_compute_forces_n_bodies<<<params.gridSize, params.blockSize>>>(x, y, z, vx, vy, vz, ax, ay, az, mass, sorted, child, left, right, p_count, gravity);
}

void UpdateParticles(float *x, float *y, float *z, float *vx, float *vy, float *vz,  float *ax, float *ay, float *az, int n, float dt, float damp)
{
	kernel6_update_velocity_position<<<params.gridSize, params.blockSize>>>(x, y, z, vx, vy, vz, ax, ay, az, n, dt, damp);	
}
void PopulateCoordinates(float *x, float *y, float *z, float *out, int p_count)
{
	aux_kernel_copy_3D_coordinate_array<<<params.gridSize, params.blockSize>>>(x, y, z, out, p_count);
}

void ResetArrays(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start, int *sorted, int *child, int *index, float *left, float *right, float *bottom, float *top, float* front, float *back,int p_count, int node_count)
{
	aux_kernel_initialize_device_arrays<<<params.gridSize, params.blockSize>>>(mutex, x, y, z, mass, count, start, sorted, child, index, left, right, bottom, top, front, back, p_count, node_count);
}


