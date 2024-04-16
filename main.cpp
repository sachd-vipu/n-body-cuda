#include <iostream>
#include <string.h>
#include <bits/stdc++.h>
#include "NBodiesSimulation.hpp"

using namespace std;


void displayDeviceProperties()
{
	// Set up CUDA device 
	cudaDeviceProp properties;

	cudaGetDeviceProperties(&properties,0);

	int fact = 1024;
	int driverVersion, runtimeVersion;

	cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

	std::cout << "************************************************************************" << std::endl;
	std::cout << "                          NVIDIA CUDA device Properties                 " << std::endl;
	std::cout << "************************************************************************" << std::endl;
	std::cout << "Name:                                    " << properties.name << std::endl;
	std::cout << "CUDA driver/runtime version:             " << driverVersion/1000 << "." << (driverVersion%100)/10 << "/" << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl;
	std::cout << "CUDA compute capabilitiy:                " << properties.major << "." << properties.minor << std::endl;
	std::cout << "Number of multiprocessors:               " << properties.multiProcessorCount << std::endl;                           
	std::cout << "GPU clock rate:                          " << properties.clockRate/fact << " (MHz)" << std::endl;
	std::cout << "Memory clock rate:                       " << properties.memoryClockRate/fact << " (MHz)" << std::endl;
	std::cout << "Memory bus width:                        " << properties.memoryBusWidth << "-bit" << std::endl;
	std::cout << "Theoretical memory bandwidth:            " << (properties.memoryClockRate/fact*(properties.memoryBusWidth/8)*2)/fact <<" (GB/s)" << std::endl;
	std::cout << "Device global memory:                    " << properties.totalGlobalMem/(fact*fact) << " (MB)" << std::endl;
	std::cout << "Shared memory per block:                 " << properties.sharedMemPerBlock/fact <<" (KB)" << std::endl;
	std::cout << "Constant memory:                         " << properties.totalConstMem/fact << " (KB)" << std::endl;
	std::cout << "Maximum number of threads per block:     " << properties.maxThreadsPerBlock << std::endl;
	std::cout << "Maximum thread dimension:                [" << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[2] << "]" << std::endl;
	std::cout << "Maximum grid size:                       [" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << "]" << std::endl;
	std::cout << "**************************************************************************" << std::endl;
	std::cout << "                                                                          " << std::endl;
	std::cout << "**************************************************************************" << std::endl;
}




int main(int argc, char** argv)
{
	int num_bodies = 256*128;  
	displayDeviceProperties();

	NBodiesSimulation simulation = new NBodiesSimulation(num_bodies);
	simulation.runAnimation();
	simulation.~NBodiesSimulation();

}