#include <cuda_runtime.h>
#include <fstream>
#include "utils.hpp"
#include <iostream>
#include <random>

using namespace std;

int calculateNumNodes(int BodyCount, int maxComputeUnits, int warpSize){
	int numberOfNodes = BodyCount * 3;
		if (numberOfNodes < 1024 * maxComputeUnits)
			numberOfNodes = 1024 * maxComputeUnits;
		// multiple of 32	
		while ((numberOfNodes & (warpSize - 1)) != 0)
			++numberOfNodes;

			return numberOfNodes;
}

void writeAsciiOutput(const char* filename, float* x, float* y, float* z, float* vx, float* vy, float* vz, int numParticles, float elapsedTime) {
    std::ofstream outFile(filename);

    if (!outFile) {
        cerr << "Error opening output file." << endl;
        return;
    }

    outFile << elapsedTime << endl; 
    outFile << numParticles << endl; 

    float radius = 1.0; 
    int ID = 1;         
    int type = 0;       
    float value1 = 1; 
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution1(0, 0.5);
	std::uniform_real_distribution<float> distribution2(0.5, 1);

    // Write data for each particle
    for (int i = 0; i < numParticles; i++) {
		if(i <= 3*numParticles/4)
			value1 = distribution1(generator);
		else	
			value1 = distribution2(generator);

        outFile << x[i] << " " << y[i] << " " << z[i] << " "
                << vx[i] << " " << vy[i] << " " << vz[i] << " "
                << radius << " " << ID << " " << type << " " << value1 << endl;
    }

    outFile.close();
}

void displayGPUProp()
{
	// Set up CUDA device 
	cudaDeviceProp properties;

	cudaGetDeviceProperties(&properties,0);

	int multiplier = 1024;
	int driverVersion, runtimeVersion;

	cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);


	std::cout << "************************************************************************" << std::endl;
	std::cout << "                          NVIDIA CUDA device Properties                 " << std::endl;
	std::cout << "************************************************************************" << std::endl;
	std::cout << "Name:                                    " << properties.name << std::endl;
	//std::cout << "CUDA driver/runtime version:             " << driverVersion/1000 << "." << (driverVersion%100)/10 << "/" << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl;
	//std::cout << "CUDA compute capabilitiy:                " << properties.major << "." << properties.minor << std::endl;
	std::cout << "Number of Compute Units (SM's):          " << properties.multiProcessorCount << std::endl;                           
	std::cout << "GPU clock :                          " << properties.clockRate/multiplier << " (MHz)" << std::endl;
	std::cout << "Memory clock :                       " << properties.memoryClockRate/multiplier << " (MHz)" << std::endl;
	std::cout << "Device global memory:                    " << properties.totalGlobalMem/(multiplier*multiplier) << " (MB)" << std::endl;
	std::cout << "Shared memory per block:                 " << properties.sharedMemPerBlock/multiplier <<" (KB)" << std::endl;
	std::cout << "Constant memory:                         " << properties.totalConstMem/multiplier << " (KB)" << std::endl;
	std::cout << "Maximum number of threads per block:     " << properties.maxThreadsPerBlock << std::endl;
	std::cout << "Maximum thread dimension:                [" << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[2] << "]" << std::endl;
	std::cout << "Maximum grid size:                       [" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << "]" << std::endl;
	std::cout << "Warp size:                               " << properties.warpSize << std::endl;
	std::cout << "**************************************************************************" << std::endl;
	std::cout << "                                                                          " << std::endl;
	std::cout << "**************************************************************************" << std::endl;

}
