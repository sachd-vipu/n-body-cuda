#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

const int blockSize = 256;
const int warp = 32;
const int stackSize = 64;
const float eps2 = 0.025;

// Ref: An Efficient CUDA Implementation of the Barnes-Hut Algorithm for the n-Body Simulation
// Ref: Section B.5, B.6  https://www.aronaldg.org/courses/compecon/parallel/CUDA_Programming_Guide_2.2.1.pdf
__global__ void kernel1_bounding_box_computation(float* x, float* y, float *z, float* top, float* bottom, float* right, float* left, float *front, float *back, int* mutex, int p_count) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

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
    for (int i = index + stride; i < p_count; i += stride) {
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


__global__ void kernel2_construct_octree(float* x, float* y, float *z, float* top, float* bottom, float* right, float* left, float *front, float *back, float* mass, int* count, int *root, int *child,  int* index, int p_count){

    int cu_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    bool isNewBody = true;

    // build the octree
    float l, r, t, b, f, ba;
    int temp, childPath;

    for(int i = cu_index ; i < p_count ;){

        if(isNewBody){
            l = *left;
            r = *right;
            t = *top;
            b = *bottom;
            f = *front;
            ba = *back;
            temp = 0;
            childPath = 0;

            // Child Path for 3D
            if(x[i] < (l + r) * 0.5){
                r = (l + r) * 0.5;
                childPath += 1;
            }
            else{
                l = (l + r) / 2;
            }

            if(y[i] < (t + b) * 0.5){
                t = (t + b) * 0.5;
                childPath += 2;
            }
            else{
                b = (t + b) * 0.5;
            }

            if(z[i] < (f + ba) * 0.5){
                f = (f + ba) * 0.5;
                childPath += 4;
            }
            else{
                ba = (f + ba) * 0.5;
            }
            isNewBody = false;
        }

        int ch_index = child[temp*8 + childPath];

        for( int ch_index = child[temp*8 + childPath]; ch_index > p_count; ch_index = child[temp*8 + childPath]){
            temp = ch_index;
            childPath = 0;

            if(x[i] < (l + r) * 0.5){
                r = (l + r) * 0.5;
                childPath += 1;
            }
            else{
                l = (l + r) / 2;
            }

            if(y[i] < (t + b) * 0.5){
                t = (t + b) * 0.5;
                childPath += 2;
            }
            else{
                b = (t + b) * 0.5;
            }

            if(z[i] < (f + ba) * 0.5){
                f = (f + ba) * 0.5;
                childPath += 4;
            }
            else{
                ba = (f + ba) * 0.5;
            }

            atomicAdd(&x[temp], mass[i] * x[i]);
            atomicAdd(&y[temp], mass[i] * y[i]);
            atomicAdd(&z[temp], mass[i] * z[i]);
            atomicAdd(&mass[temp], mass[i]);
            atomicAdd(&count[temp], 1);
        
        }

        if(ch_index != -2){
            int lock = temp * 8 + childPath;
            if(atomicCAS(&child[lock], ch_index, -2) == ch_index){
                if( ch_index == -1){
                    child[lock] = i;
                }
                else{
                    int patch = 8 * p_count;
                    while(ch_index >=0 && ch_index < p_count){
                        int cell =  atomicAdd(index, 1);
                        patch = min(patch, cell);
                        
                        if(patch != cell){
                            child[8*temp + childPath] = cell;
                        }

                        //old
                        childPath = 0;
                        if(x[ch_index] < (l + r) * 0.5){             
                            childPath += 1;
                        }
                        
                        if(y[ch_index] < (t + b) * 0.5){
                            childPath += 2;
                        }

                        if(z[ch_index] < (f + ba) * 0.5){
                            childPath += 4;
                        }

                        x[cell] += mass[ch_index] * x[ch_index];
                        y[cell] += mass[ch_index] * y[ch_index];
                        z[cell] += mass[ch_index] * z[ch_index];
                        mass[cell] += count[ch_index];
                        child[8 * cell + childPath] = ch_index;
                        root[ch_index] = -1;

                        // new

                        temp = cell;
                        childPath = 0;
                        if(x[i] < (l + r) * 0.5){
                            r = (l + r) * 0.5;
                            childPath += 1;
                        }
                        else{
                            l = (l + r) * 0.5;
                        }

                        if(y[i] < (t + b) * 0.5){
                            t = (t + b) * 0.5;
                            childPath += 2;
                        }
                        else{
                            b = (t + b) * 0.5;
                        }

                        if(z[i] < (f + ba) * 0.5){
                            f = (f + ba) * 0.5;
                            childPath += 4;
                        }
                        else{
                            ba = (f + ba) * 0.5;
                        }

                        x[cell] += mass[i] * x[i];
                        y[cell] += mass[i] * y[i];
                        z[cell] += mass[i] * z[i];
                        mass[cell] += mass[i];
                        count[cell] += count[i];
                        ch_index = child [ 8 * temp + childPath];
                    }

                    child[8 * temp + childPath] = i;
                    __threadfence();
                    child[lock] = patch;

                }

                isNewBody = true;
                i += stride;
            }
        }
        __syncthreads();
    } 
}

__global__ void kernel3_body_information_octree_node(float* x, float* y, float *z, float *mass, int *index,  int p_count) {

    int cu_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // calculate the center of mass and total mass of the node
    for(int i = cu_index + p_count ; i < *index; i += stride) {
        x[i] /= mass[i];
        y[i] /= mass[i];
        z[i] /= mass[i];
    }

}

__global__ void kernel4_approximation_sorting(int *count, int *root, int *sorted, int *child, int *index, int p_count) {

    int cu_index = p_count + threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int start = 0;

    if(threadIdx.x == 0) {
       for(int i=0;i<8;i++){
	          int node = child[i];
            if(node >= p_count){ 
                root[node] = start;
                start += count[node];
            }
            else if(node >= 0){ 
                sorted[start] = node;
                start++;
            }
        }
    }

    // for each cell i    
    for(int i = cu_index + p_count; i < *index; i += stride) {
        start = root[i];
        if(start >=0){
            for(int j = 0; j < 8; j++){
                // in-order traversal of the children
                int node = child[i*8 + j];
                if(node >= p_count){ 
                    root[node] = start;
                    start += count[node];
                }
                else if(node >= 0){ 
                    sorted[start] = node;
                    start++;
                }
            }
        }
    }

}


__global__ void kernel5_compute_forces_n_bodies(float* x, float *y, float *z,float *vx, float *vy, float *vz, float *ax, float *ay, float *az, float *mass, int *sorted, int *child, float *left, float *right, int p_count)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float depth_s[stackSize*blockSize /warp];
    __shared__ float stack_s[stackSize*blockSize /warp];

    float particle_radius = 0.5f * (right[0] - left[0]);

    // Adjust jj initialization for octree (8 children)
    int jj = -1;                 
    for (int i = 0; i < 8; i++) {       
        if (child[i] != -1) {     
            jj++;               
        }                       
    }

    int counter = threadIdx.x % warp;
    int stackStartIndex = stackSize * (threadIdx.x / warp);

    for(int i= index; i< p_count; i+= stride ){

        int sortedIndex = sorted[i];
        float pos_x = x[sortedIndex];
        float pos_y = y[sortedIndex];
        float pos_z = z[sortedIndex];
        float acl_x = 0.0f, acl_y = 0.0f, acl_z = 0.0;

        int top = jj _ stackStartIndex;
        if(counter == 0){
            int tmp = 0;
            for (int i = 0; i < 8; i++) {  // Adjust loop for 8 children
                if (child[i] != -1) {
                    stack_s[stackStartIndex + tmp] = child[i];
                    depth_s[stackStartIndex + tmp] = particle_radius * particle_radius / 0.5;
                    tmp++;
                }
            }
        }
        __syncthreads();

        for (; top >= stackSize; top--) {
        int node = stack_s[top];
        float depth = depth_s[top];
         // Loop over 8 children for octree
            for (int i = 0; i < 8; i++) {
                int ch = child[8 * node + i];  // Adjust indexing for 8 children

                if (ch >= 0) {
                    // Include the z dimension in distance calculation
                    float dx = x[ch] - pos_x;
                    float dy = y[ch] - pos_y;
                    float dz = z[ch] - pos_z;  // z difference
                    float radii = dx * dx + dy * dy + dz * dz + eps2;//(avoid div by 0);
                    if (ch < p_count  || __all(0.25* depth <= radii) ) { 
                        radii = rsqrt(radii);
                        float f = mass[ch] * radii * radii * radii;

                        acl_x += f * dx;
                        acl_y += f * dy;
                        acl_z += f * dz;  // z acceleration
                    }

                    else {
                        if (counter == 0) {
                            stack_s[top] = ch;
                            depth_s[top] = 0.25 * depth;
                        }
                        top++;
                    }
                }
            }
        }
        ax[sortedIndex] = acl_x;
        ay[sortedIndex] = acl_y;
        az[sortedIndex] = acl_z;

        __syncthreads();
  
    }
}

__global__ void kernel6_update_velocity_position(float* x, float *y, float *z,  float *vx, float *vy, float *vz, float *ax, float *ay, float *az, int p_count, float dt, float damp) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < p_count; i += stride) {
        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt; 
        vz[i] += az[i] * dt;

        x[i] += vx[i] * dt * damp;
        y[i] += vy[i] * dt * damp;
        z[i] += vz[i] * dt * damp;
    }
}

__global__ void aux_kernel_copy_3D_coordinate_array(float *x, float *y, float *z, float *output, int p_count) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < p_count; i += stride) {
        output[3*i] = x[i];
        output[3*i + 1] = y[i];
        output[3*i + 2] = z[i];
    }
}

__global__ void aux_kernel_initialize_device_arrays(float *x, float *y, float *z, float *top, float *bottom, float *right, float *left, float *front, float *back, float *mass, int *count, int *root, int* sorted, int *child, int *index, int* mutex, int p_count, int node_count) {
    int cu_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize the arrays , only once, data already in GPU. 
    for(int i = cu_index; i < node_count; i += stride) {
        
        #pragma unroll 8
        for(int j = 0; j < 8; j++) {
            child[8*i + j] = -1;
        }

        if(i < p_count) {
            count[i] = 1;
        }
        else {
            count[i] = 0;
            x[i] = 0.0f;
            y[i] = 0.0f;
            z[i] = 0.0f;
            mass[i] = 0.0f;
        }
        root[i] = -1;
        sorted[i] = -1;
    }
    
    if(cu_index == 0){
	*mutex = 0;
	*left = 0;
	*right = 0;
	*bottom = 0;
	*top = 0;
    *front = 0;
    *back = 0;   
   	*index = p_count;

    }
    
}