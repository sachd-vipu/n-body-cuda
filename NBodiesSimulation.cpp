#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include "GPU_CONFIG.hpp"
#include "NBodiesSimulation.hpp"
#include "kernel_api.cuh"
using namespace std;
#include <stdio.h>
#include <cuda.h>
#include <unistd.h>


#define cudaErrorCheck() { \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s\n", cudaGetErrorString(err)); \
	} \
}


const GLchar* vertexSource =
    "#version 130\n"
    "in vec2 position;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform mat4 projection;"
    "void main()"
    "{"
    "    gl_Position = projection * view * model *vec4(position, 0.0, 1.0);"
    "}";

const GLchar* fragmentSource =
    "#version 130\n"
    //"out vec4 outColor;"
    "void main()"
    "{"
    "    gl_FragColor = vec4(1.0, 1.0, 1.0, 0.1);"
    "}"; 

static int calculateNumNodes(int BodyCount, int maxComputeUnits, int warpSize){
	int numberOfNodes = BodyCount * 2;
		if (numberOfNodes < 1024 * maxComputeUnits)
			numberOfNodes = 1024 * maxComputeUnits;
		// multiple of 32	
		while ((numberOfNodes & (warpSize - 1)) != 0)
			++numberOfNodes;

			return numberOfNodes;
}





NBodiesSimulation::NBodiesSimulation(const int num_bodies){
			BodyCount = num_bodies;
			Nodes = calculateNumNodes(num_bodies,COMPUTE_UNITS,WARP_SIZE) + 1;

			host_left = new float;
			host_right = new float;
			host_bottom = new float;
			host_top = new float;
			host_front = new float;
			host_back = new float;

			host_x = new float[Nodes];
			host_y = new float[Nodes];
			host_z = new float[Nodes];
			host_vx = new float[Nodes];
			host_vy = new float[Nodes];
			host_vz = new float[Nodes];
			host_ax = new float[Nodes];
			host_ay = new float[Nodes];
			host_az = new float[Nodes];
			host_mass = new float[Nodes];
			host_child = new int[8 * Nodes];
			host_root = new int[Nodes];
			host_sorted = new int[Nodes];
			host_count = new int[Nodes];
			host_output = new float[Nodes * 3];

			cudaMalloc((void**)&device_left, sizeof(float));
			cudaMalloc((void**)&device_right, sizeof(float));
			cudaMalloc((void**)&device_bottom, sizeof(float));
			cudaMalloc((void**)&device_top, sizeof(float));
			cudaMalloc((void**)&device_front, sizeof(float));
			cudaMalloc((void**)&device_back, sizeof(float));
			cudaMemset(device_left, 0, sizeof(float));
			cudaMemset(device_right, 0, sizeof(float));
			cudaMemset(device_bottom, 0, sizeof(float));
			cudaMemset(device_top, 0, sizeof(float));
			cudaMemset(device_front, 0, sizeof(float));
			cudaMemset(device_back, 0, sizeof(float));

			cudaMalloc((void**)&device_x, Nodes * sizeof(float));
			cudaMalloc((void**)&device_y, Nodes * sizeof(float));
			cudaMalloc((void**)&device_z, Nodes * sizeof(float));
			cudaMalloc((void**)&device_vx, Nodes * sizeof(float));
			cudaMalloc((void**)&device_vy, Nodes * sizeof(float));
			cudaMalloc((void**)&device_vz, Nodes * sizeof(float));
			cudaMalloc((void**)&device_ax, Nodes * sizeof(float));
			cudaMalloc((void**)&device_ay, Nodes * sizeof(float));
			cudaMalloc((void**)&device_az, Nodes * sizeof(float));
			cudaMalloc((void**)&device_mass, Nodes * sizeof(float));
			cudaMalloc((void**)&device_index, sizeof(int));
			cudaMalloc((void**)&device_child, 8 * Nodes * sizeof(int));
			cudaMalloc((void**)&device_root, Nodes * sizeof(int));
			cudaMalloc((void**)&device_sorted, Nodes * sizeof(int));
			cudaMalloc((void**)&device_count, Nodes * sizeof(int));
			cudaMalloc((void**)&device_mutex, sizeof(int));

			cudaMemset(device_root, -1, Nodes * sizeof(int));
			cudaMemset(device_sorted, 0, Nodes * sizeof(int));
			cudaErrorCheck();
			cudaMalloc((void**)&device_output, Nodes * 3 * sizeof(float));

			if(PLOT_OPENGL){
				settings = new sf::ContextSettings();
				settings->depthBits = 24;
				settings->stencilBits = 8;
				window = new sf::Window(sf::VideoMode(1000, 1000, 32), "Barnes Hut SImulation", sf::Style::Titlebar | sf::Style::Close, *settings);

				glewExperimental = GL_TRUE;
				glewInit();

				// Vertex shader
				vertexShader = glCreateShader(GL_VERTEX_SHADER);
				glShaderSource(vertexShader, 1, &vertexSource, NULL);
				glCompileShader(vertexShader);

				// Fragment shader
				fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
				glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
				glCompileShader(fragmentShader);

				// Link the
				shaderProgram = glCreateProgram();
				glAttachShader(shaderProgram, vertexShader);
				glAttachShader(shaderProgram, fragmentShader);
				glBindFragDataLocation(shaderProgram, 0, "outColor");
				glLinkProgram(shaderProgram);
				glUseProgram(shaderProgram);
			}

		}

		NBodiesSimulation::~NBodiesSimulation(){
			
			cout << "Destructor called" << endl;
			// print host_output
			// for(int i=0;i<3*Nodes;i++){
			// cout << host_output[i] << " ";
			// if (i%200 == 0){
			// 	cout << endl;
			// }
			// }

			delete host_left;
			delete host_right;
			delete host_bottom;
			delete host_top;
			delete host_front;
			delete host_back;
			delete [] host_x;
			delete [] host_y;
			delete [] host_z;
			delete [] host_vx;
			delete [] host_vy;
			delete [] host_vz;
			delete [] host_ax;
			delete [] host_ay;
			delete [] host_az;
			delete [] host_mass;
			delete [] host_child;
			delete [] host_root;
			delete [] host_sorted;
			delete [] host_count;
			delete [] host_output;

	
		

			cudaFree(device_left);
			cudaFree(device_right);
			cudaFree(device_bottom);
			cudaFree(device_top);
			cudaFree(device_front);
			cudaFree(device_back);
			cudaFree(device_x);
			cudaFree(device_y);
			cudaFree(device_z);
			cudaFree(device_vx);
			cudaFree(device_vy);
			cudaFree(device_vz);
			cudaFree(device_ax);
			cudaFree(device_ay);
			cudaFree(device_az);
			cudaFree(device_mass);
			cudaFree(device_child);
			cudaFree(device_root);
			cudaFree(device_sorted);
			cudaFree(device_count);
			cudaFree(device_index);
			cudaFree(device_mutex);
			cudaFree(device_output);
			
			cudaError_t err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
					printf("CUDA error: %s\n", cudaGetErrorString(err));
				}
			cudaErrorCheck();	

			if(PLOT_OPENGL){
				delete settings;
				delete window;

				glDeleteProgram(shaderProgram);
				glDeleteShader(fragmentShader);
				glDeleteShader(vertexShader);
			}
			cout << "Destructor finished" << endl;
		}




		void NBodiesSimulation::runAnimation()
		{
		setParticlePosition(host_x, host_y, host_z, host_vx, host_vy, host_vz, host_ax, host_ay, host_az, host_mass, BodyCount);
	cudaMemcpy(device_mass, host_mass, 2*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_x, host_x, 2*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_y, host_y, 2*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_z, host_z, 2*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_vx, host_vx, 2*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_vy, host_vy, 2*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_ax, host_ax, 2*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_ay, host_ay, 2*BodyCount*sizeof(float), cudaMemcpyHostToDevice);

	
		for(int i=0;i< ITERATIONS ;i++){ 
			float time;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			ResetArrays(device_mutex, device_x, device_y, device_z, device_mass, device_count, device_root, device_sorted, device_child, device_index, device_left, device_right, device_bottom, device_top, device_front, device_back, BodyCount, Nodes);
			ComputeBoundingBox(device_mutex, device_x, device_y, device_z,  device_left, device_right, device_bottom, device_top,  device_front, device_back,  BodyCount);
			ConstructOctree(device_x, device_y, device_z, device_mass, device_count, device_root, device_child, device_index, device_left, device_right, device_bottom, device_top, device_front, device_back, BodyCount);
			ComputeBodyInfo(device_x, device_y, device_z, device_mass, device_index, BodyCount);
			SortBodies(device_count, device_root, device_sorted, device_child, device_index, BodyCount);
			CalculateForce(device_x, device_y, device_z, device_vx, device_vy, device_vz, device_ax, device_ay, device_az, device_mass, device_sorted, device_child, device_left, device_right, BodyCount, GRAVITY);
			UpdateParticles(device_x, device_y, device_z, device_vx, device_vy, device_vz, device_ax, device_ay, device_az, BodyCount, TIMESTEP, DAMP);
			PopulateCoordinates(device_x, device_y, device_z, device_output, Nodes);

			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			cudaEventDestroy(stop);
			cout << "Time taken for iteration " <<  i << " is " << time << endl;

			if(PLOT_OPENGL){
			cudaMemcpy(host_output, device_output, 2*Nodes*sizeof(float), cudaMemcpyDeviceToHost);
					//cudaDeviceSynchronize();
			    const float* vertices = getOutput();

				glGenVertexArrays(1, &vao);
				glBindVertexArray(vao);

				glGenBuffers(1, &vbo);   //generate a buffer
				glBindBuffer(GL_ARRAY_BUFFER, vbo);   //make buffer active
				glBufferData(GL_ARRAY_BUFFER, 3 * BodyCount *sizeof(float), vertices, GL_DYNAMIC_DRAW); //copy data to active buffer 

				// Specify the layout of the vertex data
				GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
				glEnableVertexAttribArray(posAttrib);
				glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);

				glBlendFunc(GL_SRC_ALPHA, GL_ONE);
				glEnable(GL_BLEND);

				// model, view, and projection matrices
				glm::mat4 model = glm::mat4(1.0f);
				glm::mat4 view = glm::mat4(1.0f);
				// view = glm::rotate(view, float(2*i), glm::vec3(0.0f, 1.0f, 0.0f)); 
				glm::mat4 projection = glm::ortho(-25.0f, 25.0f, -25.0f, 25.0f, -10.0f, 10.0f);

				// link matrices with shader program
				GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
				GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
				GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
				glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

				// Clear the screen to black
				glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
				glClear(GL_COLOR_BUFFER_BIT);

				// Draw points
				glDrawArrays(GL_POINTS, 0, BodyCount);

				// Swap buffers
				window->display();

				glDeleteBuffers(1, &vbo);

				glDeleteVertexArrays(1, &vao);
				
			}

		}
		
		if(PLOT_OPENGL){
			window->close(); 
		}
	}

		const float* NBodiesSimulation::getOutput()
		{
			return host_output;
		}




void NBodiesSimulation::setParticlePosition(float* x, float* y, float* z, float* vx, float* vy, float* vz,  float* ax, float*ay, float*az, float* mass, float p_count)
{
			
	float pi = 3.14159265;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution1(1.5, 12.0);
	std::uniform_real_distribution<float> distribution2(1, 5.0);
	std::uniform_real_distribution<float> distribution_theta(0.0, 2 * pi);


	// loop through all particles
	for (int i = 0; i < p_count; i++){
		float theta = distribution_theta(generator);
		float r1 = distribution1(generator);
		float r2 = distribution2(generator);

		// set mass and position of particle
		if(i==0){
			mass[i] = 100000;
			x[i] = 0;
			y[i] = 0;
			z[i] = 0;
		}
		else if(i==1){
			mass[i] = 25000;
			x[i] = 20*cos(theta);
			y[i] = 20*sin(theta);
			z[i] = (x[i] + y[i] ) *0.5;

		}
		else if(i<=3*p_count/4){
			mass[i] = 1.0;
			x[i] = r1*cos(theta);
			y[i] = r1*sin(theta);
						z[i] = (x[i] + y[i] ) *0.5;

		}
		else{
			mass[i] = 1.0;
			x[i] = r2*cos(theta) + x[1];
			y[i] = r2*sin(theta) + y[1];
			z[i] = (x[i] + y[i] ) *0.5;

		}


		// set velocity of particle
		float rotation = 1;  // 1: clockwise   -1: counter-clockwise 
		float v1 = 1.0*sqrt(GRAVITY*100000.0 / r1);
		float v2 = 1.0*sqrt(GRAVITY*25000.0 / r2);
		float v = 1.0*sqrt(GRAVITY*100000.0 / sqrt(800));
		if(i==0){
			vx[0] = 0;
			vy[0] = 0;
			vz[0] = 0;
		}
		else if(i==1){
			vx[i] = 0.0;//rotation*v*sin(theta);
			vy[i] = 0.0;//-rotation*v*cos(theta);
			vz[i] = 0.0;
		}
		else if(i<=3*p_count/4){
			vx[i] = rotation*v1*sin(theta);
			vy[i] = -rotation*v1*cos(theta);
			vz[i] = 0.0;
		}
		else{
			vx[i] = rotation*v2*sin(theta);
			vy[i] = -rotation*v2*cos(theta);	
			vz[i] = 0.0;		
		}

		// set acceleration to zero
		ax[i] = 0.0;
		ay[i] = 0.0;
		az[i] = 0.0;
		}

		}
		
 


	
// Use arrays instead of array of struct to maximize coalescing
// struct Position{
//     float x;
//     float y;
//     float z;
// };

// struct Velocity{
//     float vx;
//     float vy;
//     float vz;
// };

// struct Body{
//     Position position;
//     Velocity velocity;
//     float mass;
// };

// struct Node{
//     float center_of_mass;
//     float accumulatedMass;
//     float minCorner;
//     float maxCorner;
//     Node* children[8];
//     Body* body;
// };

// struct Oct_Tree
// {
//     Node* root;
//     int maxDepth;
//     int maxBodies;
//     float theta;
// };
