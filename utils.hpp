
#define cudaErrorCheck() { \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s\n", cudaGetErrorString(err)); \
	} \
}


 int calculateNumNodes(int BodyCount, int maxComputeUnits, int warpSize);

 void writeAsciiOutput(const char* filename, float* x, float* y, float* z, float* vx, float* vy, float* vz, int numParticles, float elapsedTime);

 void displayGPUProp();
 
