#include <stdio.h>
#include "barneshut_kernel.cuh"

__device__ const int blockSize = 1024;
__device__ const int warp = 32;
__device__ const int MAX_DEPTH = 128;
__device__ const float eps = 0.025;

// Ref: An Efficient CUDA Implementation of the Barnes-Hut Algorithm for the n-Body Simulation
// Ref: Section B.5, B.6  https://www.aronaldg.org/courses/compecon/parallel/CUDA_Programming_Guide_2.2.1.pdf  

__global__ void kernel1_bounding_box_computation(int *mutex, float *x, float *y, float *z, float *left, float *right, float *bottom, float *top, float *front, float* back, int p_count)
{
	int cu_index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	float x_min, x_max,y_min,y_max, z_min, z_max;
	x_min = x_max = x[cu_index];
	y_min = y_max = y[cu_index];
	z_min = z_max = z[cu_index];
	
	__shared__ float left_s[blockSize];
	__shared__ float right_s[blockSize];
	__shared__ float top_s[blockSize];
	__shared__ float bottom_s[blockSize];
	__shared__ float front_s[blockSize];
	__shared__ float back_s[blockSize];

    // Find the bounding box for the current block
    for (int i = cu_index + stride; i < p_count; i += stride) {
		x_min = fminf(x_min, x[i]);
		x_max = fmaxf(x_max, x[i]);
		y_min = fminf(y_min, y[i]);
		y_max = fmaxf(y_max, y[i]);
		z_min = fminf(z_min, z[i]);
		z_max = fmaxf(z_max, z[i]);

	}

    // Store the bounding box in shared memory
	left_s[threadIdx.x] = x_min;
	right_s[threadIdx.x] = x_max;
	top_s[threadIdx.x] = y_max;
	bottom_s[threadIdx.x] = y_min;
	front_s[threadIdx.x] = z_max;
	back_s[threadIdx.x] = z_min;

	__syncthreads();
	
	for(int s = blockDim.x/2; s != 0; s = s >> 2){
		if(threadIdx.x < s){
			left_s[threadIdx.x] = fminf(left_s[threadIdx.x], left_s[threadIdx.x + s]);
			right_s[threadIdx.x] = fmaxf(right_s[threadIdx.x], right_s[threadIdx.x + s]);
			bottom_s[threadIdx.x] = fminf(bottom_s[threadIdx.x], bottom_s[threadIdx.x + s]);
			top_s[threadIdx.x] = fmaxf(top_s[threadIdx.x], top_s[threadIdx.x + s]);
			front_s[threadIdx.x] = fmaxf(front_s[threadIdx.x], front_s[threadIdx.x + s]);
			back_s[threadIdx.x] = fminf(back_s[threadIdx.x], back_s[threadIdx.x + s]);
		}
		__syncthreads();
	}

	if(threadIdx.x == 0){
		while (atomicCAS(mutex, 0 ,1) != 0); 
		*left = fminf(*left, left_s[0]);
		*right = fmaxf(*right, right_s[0]);
		*bottom = fminf(*bottom, bottom_s[0]);
		*top = fmaxf(*top, top_s[0]);
		*front = fmaxf(*front, front_s[0]);
		*back = fminf(*back, back_s[0]);
		atomicExch(mutex, 0);
	}
}


__global__ void kernel2_construct_octree(float *x, float *y, float *z, float *mass, int *count, int *root, int *child, int *index, float *left, float *right, float *bottom, float *top, float*front, float *back, int p_count)
{
	int cu_index = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	bool insert_success = true;

   // build the octree
   // TODO - DPETH CALCULATION to be used in kernel 4
   float l, r, t, b, f, ba;
   int temp, childPath;


	for(int i = cu_index; i < p_count;){

		if(insert_success){
			insert_success = false;

			l = *left;
			r = *right;
			b = *bottom;
			t = *top;
			f = *front;
			ba = *back;

			temp = 0;
			childPath = 0;
			if(x[i] < 0.5*(l+r)){
				childPath += 1;
				r = 0.5*(l+r);
			}
			else{
				l = 0.5*(l+r);
			}
			if(y[i] < 0.5*(b+t)){
				childPath += 2;
				t = 0.5*(t+b);
			}
			else{
				b = 0.5*(t+b);
			}

			if(z[i] < 0.5*(f+ba)){
				childPath += 4;
				f = 0.5*(f+ba);
			}
			else{
				ba = 0.5*(f+ba);
			}
		}
		int ch_index = child[temp*8 + childPath];

		//we go only over cells
		for(;ch_index >= p_count;){
			temp = ch_index;
			childPath = 0;
			if(x[i] < 0.5*(l+r)){
				childPath += 1;
				r = 0.5*(l+r);
			}
			else{
				l = 0.5*(l+r);
			}
			if(y[i] < 0.5*(b+t)){
				childPath += 2;
				t = 0.5*(t+b);
			}
			else{
				b = 0.5*(t+b);
			}
			if(z[i] < 0.5*(f+ba)){
				childPath += 4;
				f = 0.5*(f+ba);
			}
			else{
				ba = 0.5*(f+ba);
			}

			atomicAdd(&x[temp], mass[i]*x[i]);
			atomicAdd(&y[temp], mass[i]*y[i]);
			atomicAdd(&z[temp], mass[i]*z[i]);
			atomicAdd(&mass[temp], mass[i]);
			atomicAdd(&count[temp], 1);
			ch_index = child[8*temp + childPath];
		}


		if(ch_index != -2){
			int lock = temp * 8 + childPath;
			if(atomicCAS(&child[lock], ch_index, -2) == ch_index){
				if(ch_index == -1){
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

				 		childPath = 0;
				 		if(x[ch_index] < 0.5*(l+r)){
				 			childPath += 1;
                     	}
				 		if(y[ch_index] < 0.5*(b+t)){
				 			childPath += 2;
						}
						if(z[ch_index] < 0.5*(f+ba)){
							childPath += 4;
					   }
				 		
				 			if(cell >= 2*p_count){
				 				printf("cell: %d\n", cell);
									break;
				 			}
				 		x[cell] += mass[ch_index]*x[ch_index];
				 		y[cell] += mass[ch_index]*y[ch_index];						
						z[cell] += mass[ch_index]*z[ch_index];
				 		mass[cell] += mass[ch_index];
				 		count[cell] += count[ch_index];
				 		child[8 * cell + childPath] = ch_index;
				 		root[cell] = -1;
					

				 		temp = cell;
				 		childPath = 0;
				 		if(x[i] < 0.5*(l+r)){
				 			childPath += 1;
				 			r = 0.5*(l+r);
				 		}
				 		else{
				 			l = 0.5*(l+r);
				 		}
				 		if(y[i] < 0.5*(b+t)){
				 			childPath += 2;
				 			t = 0.5*(t+b);
				 		}
				 		else{
				 			b = 0.5*(t+b);
				 		}
						if(z[i] < 0.5*(f+ba)){
							childPath += 4;
							f = 0.5*(f+ba);
						}
						else{
							ba = 0.5*(f+ba);
						}
				 		x[cell] += mass[i]*x[i];
				 		y[cell] += mass[i]*y[i];
						z[cell] += mass[i]*z[i];
				 		mass[cell] += mass[i];
				 		count[cell] += count[i];
				 		ch_index = child[8*temp + childPath]; 
				 	}
				 	child[8*temp + childPath] = i;
					 __threadfence(); 
				 	child[lock] = patch;
				}

				i += stride;
				insert_success = true;
			}

		}

		__syncthreads(); 
	}
}




__global__ void kernel3_body_information_octree_node(float *x, float *y, float *z, float *mass, int *index, int p_count)
{
	int cu_index = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;

	// cells and bodies are in same array so skip to startv calculation from the cells
	for(int i = cu_index + p_count; i < *index; i+=stride){
		x[i] /= mass[i];
		y[i] /= mass[i];
		z[i] /= mass[i]; 

	}
}



__global__ void kernel4_approximation_sorting(int *count, int *root, int *sorted, int *child, int *index, int p_count)
{
	int cu_index = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;

	int start = 0;
	if(threadIdx.x == 0){
		for(int j=0;j<8;j++){
			int node = child[j];

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

	for( int i = cu_index + p_count ; i < *index; ){
		start = root[i];
		if(start >=0){
			for(int j = 0; j < 8; j++){
				int node = child[8*i + j];
				if(node >= p_count){  
					root[node] = start;
					start += count[node];
				}
				else if(node >= 0){  
					sorted[start] = node;
					start++;
				}
			}
			i += stride;
		}
	}
}



__global__ void kernel5_compute_forces_n_bodies(float* x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay, float *az, float *mass, int *sorted, int *child, float *left, float *right, int p_count, float g){
	
	int cu_index = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;

	__shared__ float depth_s[MAX_DEPTH*blockSize/warp]; 
	__shared__ int stack_s[MAX_DEPTH*blockSize/warp];  

	float particle_radius = 0.5f *(right[0] - left[0]);

	int jj = -1;                 
	for(int i=0;i<8;i++){       
		if(child[i] != -1){     
			jj++;               
		}                       
	}

	int counter = threadIdx.x % warp;
	int stackStartIndex = MAX_DEPTH * (threadIdx.x / warp);

	for(int i= cu_index; i< p_count; i+= stride ){

		int sortedIndex = sorted[i];
		float pos_x = x[sortedIndex];      
		float pos_y = y[sortedIndex];    
		float pos_z = z[sortedIndex];  
		float acc_x = 0;
		float acc_y = 0; 
		float acc_z = 0;

		int top = jj + stackStartIndex;
		if(counter == 0){
			int temp = 0;
			for(int j = 0 ;j < 8;j++){
				if(child[j] != -1){
					stack_s[stackStartIndex + temp] = child[j];
					depth_s[stackStartIndex + temp] = particle_radius*particle_radius/0.5; // theta = 0.5
					temp++;
				}

			}
		}


		__syncthreads();

		while(top >= stackStartIndex){
			int node = stack_s[top];
			float dp = 0.25*depth_s[top];
			for(int j=0;j<8;j++){
				int ch = child[8*node + j];

			
				if(ch >= 0){
                    // Include z dimension
					float dx = x[ch] - pos_x;
  					float dy = y[ch] - pos_y;
					float dz = z[ch] - pos_z;		
    				float r = dx*dx + dy*dy +  + dz*dz + eps;
    				if(ch < p_count  || __all(dp <= r)){ 
    					r = rsqrt(r);
    					float f = mass[ch] * r * r * r;

    					acc_x += f*dx;
    					acc_y += f*dy;
						acc_z += f*dz;
					}
					else{
						if(counter == 0){
							stack_s[top] = ch;
							depth_s[top] = dp;
						}
						top++;
					}
				}
			}

			top--;
		}
		ax[sortedIndex] = acc_x;     
   		ay[sortedIndex] = acc_y;
		az[sortedIndex] = acc_z;     

   		__syncthreads();

   	}
}

__global__ void kernel6_update_velocity_position(float* x, float *y, float *z,  float *vx, float *vy, float *vz, float *ax, float *ay, float *az, int p_count, float dt, float damp) {
	int cu_index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = cu_index; i < p_count; i += stride) {
        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt; 
        vz[i] += az[i] * dt;

        x[i] += vx[i] * dt * damp;
        y[i] += vy[i] * dt * damp;
        z[i] += vz[i] * dt * damp;
    } 
}

__global__ void aux_kernel_copy_3D_coordinate_array(float *x, float *y, float *z, float *output, int p_count)
{
	int cu_index = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = cu_index; i < p_count; i += stride){
		output[3*i] = x[i];
		output[3*i + 1] = y[i];
		output[3*i + 2] = z[i];
	}
}

__global__ void aux_kernel_initialize_device_arrays(int *mutex, float *x, float *y, float* z, float *mass, int *count, int *start, int *sorted, int *child, int *index, float *left, float *right, float *bottom, float *top, float*front, float* back,int p_count, int node_count)
{
    int cu_index = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;

	// Initialize the arrays, only once, data already in GPU
	for(int i = cu_index ;i < node_count; i+=stride){  
		#pragma unroll 8
		for(int j = 0;j < 8; j++) {
			child[i*8 + j] = -1;
		}
		if(i < p_count){
			count[i] = 1;
		}
		else{
			x[i] = 0;
			y[i] = 0;
			z[i] = 0;
			mass[i]= 0;
			count[i] = 0;
		}
		start[i] = -1;
		sorted[i] = 0;
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
