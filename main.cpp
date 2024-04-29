#include <iostream>
#include <string.h>
#include "NBodiesSimulation.hpp"

using namespace std;

int main(int argc, char** argv)
{
	constexpr int GRID_SIZE = 1024;
	constexpr int BLOCK_SIZE = 128;
	int num_bodies = GRID_SIZE * BLOCK_SIZE;  
	NBodiesSimulation simulation(num_bodies);
	simulation.runAnimation();
	//simulation.~NBodiesSimulation();
	return 0;

}
