#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "Oct_Tree.h"
#define N 1000 
#define FPS 0.01f 
#define SOFTENING 1e-9f // prevents division by zero
const int blockSize = 32;

__global__ void updateBodies(Body* bodies) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (int j = 0; j < N; j++) {
            float dx = bodies[j].position.x - bodies[i].position.x;
            float dy = bodies[j].position.y - bodies[i].position.y;
            float dz = bodies[j].position.z - bodies[i].position.z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        bodies[i].velocity.vx += FPS*Fx; bodies[i].velocity.vy += FPS*Fy; bodies[i].velocity.vz += FPS*Fz;
    }
}

__global__ void integrateBodies(Body* bodies) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        bodies[i].position.x += bodies[i].velocity.vx*FPS;
        bodies[i].position.y += bodies[i].velocity.vy*FPS;
        bodies[i].position.z += bodies[i].velocity.vz*FPS;
    }
}

// Ref: An Efficient CUDA Implementation of the Barnes-Hut Algorithm for the n-Body Simulation
// Ref: Section B.5, B.6  https://www.aronaldg.org/courses/compecon/parallel/CUDA_Programming_Guide_2.2.1.pdf
__global__ void kernel1_bounding_box_computation(float* x, float* y, float *z, float* top, float* bottom, float* right, float* left, float *front, float *back, int* mutex, int p_count) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = stride;

    float x_min,x_max;
    x_min = x_max  = x[index];
    float y_min, y_max;
    y_min = y_max = y[index];
    float z_min, z_max;
    z_min = z_max = z[index];
    

    __shared__ float left_s[blockSize];
    __shared__ float right_s[blockSize];
    __shared__ float top_s[blockSize];
    __shared__ float bottom_s[blockSize];
    __shared__ float front_s[blockSize];
    __shared__ float back_s[blockSize];

    // Find the bounding box for the current block
    for (int i = index + offset; i < p_count; i += stride) {
        if (x[i] < x_min) 
            x_min = x[i];
        if (x[i] > x_max) 
            x_max = x[i];
        if (y[i] < y_min) 
            y_min = y[i];
        if (y[i] > y_max) 
            y_max = y[i];
        if (z[i] < z_min)
            z_min = z[i];
        if (z[i] > z_max)
            z_max = z[i];
    }

    // Store the bounding box in shared memory
    left_s[threadIdx.x] = x_min;
    right_s[threadIdx.x] = x_max;
    top_s[threadIdx.x] = y_max;
    bottom_s[threadIdx.x] = y_min;
    front_s[threadIdx.x] = z_max;
    back_s[threadIdx.x] = z_min;

    __syncthreads();


    // Reduce the bounding box to find the bounding box for the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        
        if (threadIdx.x < s) {
            if (left_s[threadIdx.x + s] < left_s[threadIdx.x]) 
                left_s[threadIdx.x] = left_s[threadIdx.x + s];

            if (right_s[threadIdx.x + s] > right_s[threadIdx.x]) 
                right_s[threadIdx.x] = right_s[threadIdx.x + s];

            if (top_s[threadIdx.x + s] > top_s[threadIdx.x]) 
                top_s[threadIdx.x] = top_s[threadIdx.x + s];
                
            if (bottom_s[threadIdx.x + s] < bottom_s[threadIdx.x]) 
                bottom_s[threadIdx.x] = bottom_s[threadIdx.x + s];
            
            if (front_s[threadIdx.x + s] > front_s[threadIdx.x])
                front_s[threadIdx.x] = front_s[threadIdx.x + s];
            
            if (back_s[threadIdx.x + s] < back_s[threadIdx.x])
                back_s[threadIdx.x] = back_s[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Assign the writing block bounding box to global memory to thread 0
    if (threadIdx.x == 0) {
        while (atomicCAS(mutex, 0, 1) != 0);
        if (left_s[0] < *left) 
            *left = left_s[0];
        if (right_s[0] > *right) 
            *right = right_s[0];
        if (top_s[0] > *top) 
            *top = top_s[0];
        if (bottom_s[0] < *bottom) 
            *bottom = bottom_s[0];
        if (front_s[0] > *front)
            *front = front_s[0];
        if (back_s[0] < *back) 
            *back = back_s[0];
            
        atomicExch(mutex, 0);
    }


}


__global__ void kernel2_construct_octree(float* x, float* y, float *z, float* top, float* bottom, float* right, float* left, float *front, float *back, float* mass, int* count, int *root, int *child,  int p_count){

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    // build the octree



}

__global__ void kernel3_body_information_octree_node(float* x, float* y, float *z, float *mass, int *index,  int p_count) {

    int cu_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    // calculate the center of mass and total mass of the node
    for(int i = cu_index + offset; i < *index; i += stride) {
        x[i] /= mass[i];
        y[i] /= mass[i];
        z[i] /= mass[i];
    }

}


int main(int argc, char** argv) {
    Body* bodies;
    return 0;
}