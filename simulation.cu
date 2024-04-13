#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "Oct_Tree.h"
#define N 1000 
#define FPS 0.01f 
#define SOFTENING 1e-9f // prevents division by zero


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

int main(int argc, char** argv) {
    Body* bodies;
    return 0;
}