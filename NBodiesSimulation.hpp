#include <cuda_runtime.h>


class NBodiesSimulation
{
		int BodyCount;
		int Nodes;

        // the sequence in which these arrays will be populated is important to maximize coalescing
       
       
		float *host_left;
		float *host_right;
		float *host_bottom;
		float *host_top;
        float *host_front;
        float *host_back;
	
        float *host_x;
        float *host_y;
        float *host_z;
        float *host_vx;
        float *host_vy;
        float *host_vz;
        float *host_ax;
        float *host_ay;
        float *host_az;

    	float *host_mass;

		int *host_child;
        int *host_root;
        int *host_sorted;
        int *host_count;

		float *device_left;
		float *device_right;
		float *device_bottom;
		float *device_top;
		float *device_front;
		float *device_back;
		
        float *device_x;
        float *device_y;
        float *device_z;
        float *device_vx;
        float *device_vy;
        float *device_vz;
        float *device_ax;
        float *device_ay;
        float *device_az;
        float *device_mass;

		int *device_index;
		int *device_child;
		int *device_root;
		int *device_sorted;
		int *device_count;
		int *device_mutex;  

		cudaEvent_t start, stop; 

		float *host_output;  
		float *device_output;  

	public:
	NBodiesSimulation(const int num_bodies);
	~NBodiesSimulation();
	const float* getOutput();
	void setParticlePosition(float* x, float* y, float* z, float* vx, float* vy, float* vz, float* ax, float*ay, float*az, float* mass, float p_count);			
	void runAnimation();

};
