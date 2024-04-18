#ifndef __KERNEL_API__
#define __KERNEL_API__


struct 
{
    dim3 gridSize = 512;
    dim3 blockSize = 256;
} params;  



void ComputeBoundingBox(float* x, float* y, float *z, float* top, float* bottom, float* right, float* left, float *front, float *back, int* mutex, int p_count);

void ConstructOctree(float* x, float* y, float *z, float* top, float* bottom, float* right, float* left, float *front, float *back, float* mass, int* count, int *root, int *child,  int* index, int p_count);

void ComputeBodyInfo(float* x, float* y, float *z, float *mass, int *index,  int p_count);

void SortBodies(int *count, int *root, int *sorted, int *child, int *index, int p_count);

void CalculateForce(float* x, float *y, float *z,float *vx, float *vy, float *vz, float *ax, float *ay, float *az, float *mass, int *sorted, int *child, float *left, float *right, int p_count);

void UpdateParticles(float* x, float *y, float *z,  float *vx, float *vy, float *vz, float *ax, float *ay, float *az, int p_count, float dt, float damp);

void PopulateCoordinates(float *x, float *y, float *z, float *output, int p_count);

 void ResetArrays(float *x, float *y, float *z, float *top, float *bottom, float *right, float *left, float *front, float *back, float *mass, int *count, int *root, int* sorted, int *child, int *index, int* mutex, int p_count, int node_count);

#endif