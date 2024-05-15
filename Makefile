# Makefile for nbody code using CUDA and C++

# Compiler settings
NVCC = nvcc             
CPP = g++                
OPT = -O2 -g -G         
STD = -std=c++11


# Objects and libraries
OBJECTS= main.o barneshut_kernel.o NBodiesSimulation.o kernel_api.o utils.o
LIBS= -lGL -lGLEW -lsfml-graphics -lsfml-window -lsfml-system
INCLUDES = -I/home/vipul/Desktop/n-body-cuda/glm-0.9.8.5/glm
EXEC= main

# Build target
build: $(OBJECTS)
	$(NVCC) $(OPT) -o $(EXEC) $(OBJECTS) $(LIBS) 

barneshut_kernel.o: barneshut_kernel.cu
	$(NVCC) $(OPT)  -c barneshut_kernel.cu
NBodiesSimulation.o: NBodiesSimulation.cpp
	$(NVCC) $(OPT)  $(STD) -c $(INCLUDES)  NBodiesSimulation.cpp 
kernel_api.o: kernel_api.cu
	$(NVCC) $(OPT)  -c kernel_api.cu 
utils.o: utils.cpp
	$(NVCC) $(OPT)  $(STD) -c $(INCLUDES) utils.cpp
main.o: main.cpp 
	$(NVCC) $(OPT) -c $(INCLUDES) main.cpp


.PHONY: clean
clean:
	rm -f $(OBJECTS) $(EXEC)

profile: $(EXEC)
	nsys profile --stats=true ./$(EXEC)

