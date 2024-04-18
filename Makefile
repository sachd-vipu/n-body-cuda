
	
# Makefile for nbody code using CUDA and C++

# Compiler settings
NVCC = nvcc
CPP = g++
OPT = -O2 -g -G
STD = -std=c++11

# Executable name
EXECNAME = main

# Objects and libraries
OBJECTS =  kernel_api.o NBodiesSimulation.o main.o #simulation.o
LIBS= #
INCLUDES = # Dinclude directories 

# Build target
build: $(OBJECTS)
	$(NVCC) $(OPT) -o $(EXECNAME) $(OBJECTS) $(LIBS)

# Object files

#simulation.o: simulation.cu 
#	$(NVCC) $(OPT)  -c  simulation.cu
kernel_api.o: kernel_api.cu 
	$(NVCC) $(OPT) -c kernel_api.cu 
NBodiesSimulation.o: NBodiesSimulation.cpp 
	$(NVCC) $(OPT) $(STD) -c NBodiesSimulation.cpp 
main.o: main.cpp
	$(NVCC) $(OPT) $(STD) -c $(INCLUDES)  main.cpp
# Clean target
.PHONY: clean
clean:
	rm -f $(OBJECTS) $(EXECNAME)

