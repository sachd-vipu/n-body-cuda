#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include "GPU_CONFIG.hpp"
#include "NBodiesSimulation.hpp"
#include "utils.hpp"
#include "kernel_api.cuh"
#include <stdio.h>
#include <cuda.h>
#include <unistd.h>

using namespace std;


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



NBodiesSimulation::NBodiesSimulation(const int num_bodies){
			BodyCount = num_bodies;
			Nodes = calculateNumNodes(num_bodies,COMPUTE_UNITS,WARP_SIZE);

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
		displayGPUProp();
		setParticlePosition(host_x, host_y, host_z, host_vx, host_vy, host_vz, host_ax, host_ay, host_az, host_mass, BodyCount);
		cudaMemcpy(device_mass, host_mass, 3*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(device_x, host_x, 3*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(device_y, host_y, 3*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(device_z, host_z, 3*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(device_vx, host_vx, 3*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(device_vy, host_vy, 3*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(device_ax, host_ax, 3*BodyCount*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(device_ay, host_ay, 3*BodyCount*sizeof(float), cudaMemcpyHostToDevice);

		float timeStep = 0.0;
		
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

			if(GENERATE_PARTICLEDATA){
				timeStep += time;
				cudaMemcpy(host_x, device_x, 3*BodyCount*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(host_y, device_y, 3*BodyCount*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(host_z, device_z, 3*BodyCount*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(host_vx, device_vx,3*BodyCount*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(host_vy, device_vy, 3*BodyCount*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(host_vz, device_vz, 3*BodyCount*sizeof(float), cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				char filename[50];
				sprintf(filename, "collidingDisk%04d.dat", i);
				writeAsciiOutput(filename, host_x, host_y, host_z, host_vx, host_vy, host_vz, BodyCount, timeStep);
			}

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
	collidingDisks(x, y, z, vx, vy, vz, ax, ay, az, mass, p_count);
	//spatialCube(x, y, z, vx, vy, vz, ax, ay, az, mass, p_count);
}
		
void NBodiesSimulation::collidingDisks(float* x, float* y, float* z, float* vx, float* vy, float* vz, float* ax, float*ay, float*az, float* mass, float p_count)
{
	float pi = PI;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution1(3.5, 10.0);
	std::uniform_real_distribution<float> distribution2(1.5, 5.5);

	std::uniform_real_distribution<float> distribution_theta(0.0, 2*pi);
	std::uniform_real_distribution<float> distribution_phi(0.0, 3*pi);
	std::uniform_real_distribution<float> distribution_r_disc(3.5, 12.0);
	std::uniform_real_distribution<float> distribution_z_disc(-1.0, 1.0);

	int galaxy2 = 3*p_count/4;
	// loop through all particles
	for (int i = 0; i < p_count; i++){
		float theta = distribution_theta(generator);
		float phi = distribution_phi(generator);
		float r1 = distribution1(generator);
		float r2 = distribution2(generator);
		float r_disc = distribution_r_disc(generator);
		float z_disc = distribution_z_disc(generator);
		const int limit_centre = 0;
		// set mass and position of particle
		if(i==0){
			mass[i] = 100000;
			x[i] = 0;
			y[i] = 0;
			z[i] = 0;
		}
		else if(i==galaxy2){
			mass[i] = 55000;
			x[i] = 15*sin(theta) * cos(phi);
			y[i] = 20*sin(theta) * sin(phi);
			z[i] = 20*cos(theta);

		}

		else if(i < galaxy2){
					if( i < limit_centre){
					mass[i] = 1.0;
					x[i] = 3.5*sin(theta) * cos(phi);
					y[i] = 3.5*sin(theta) * sin(phi);
					z[i] = 3.5*cos(theta);
					}
				else {  
						mass[i] = 1.0;
						x[i] = r_disc * cos(theta);  
						y[i] = r_disc * sin(theta);
						z[i] = z_disc * (1 - (r_disc - 3.5) / 8.5) * sin(phi) * cos(phi); // TO MAKE THE DISK GALAXY THINNER AT EDGES
						}

		}
	else{
		mass[i] = 2.0;

		float x_original = r2 * cos(theta);
		float y_original = r2 * sin(theta);
		float z_original =  z_disc * (1 - (r2 - 1.5) / 4.0); // TO MAKE THE DISK GALAXY THINNER AT EDGES

		float x_rotatedevice_z = cos(pi/3) * x_original - sin(pi/3) * y_original;
		float y_rotatedevice_z = sin(pi/3) * x_original + cos(pi/3) * y_original;
		float z_rotatedevice_z = z_original;

		float x_final = x_rotatedevice_z;
		float y_final = cos(pi/6) * y_rotatedevice_z - sin(pi/6) * z_rotatedevice_z;
		float z_final = sin(pi/6) * y_rotatedevice_z + cos(pi/6) * z_rotatedevice_z;

		x[i] =  x_final +  x[galaxy2];
		y[i] =  y_final + y[galaxy2];
		z[i] =  z_final + z[galaxy2];

		}

		float rotation = 1;  
		float v1 = 1.0*sqrt(GRAVITY*100000.0 / r1);
		float v2 = 1.0*sqrt(GRAVITY*55000.0 / r2);
		if(i==0 || i == 1){
			vx[i] = 0;
			vy[i] = 0;
			vz[i] = 0;
		}
		else if(i< galaxy2){
			if (i < limit_centre){
				vx[i] = 0;
				vy[i] = -1 * rotation*v1*cos(theta);
				vz[i] = rotation*v1*sin(theta);
			}
			else{
				vx[i] = rotation*v1*sin(theta) ;
				vy[i] = -1 * rotation*v1*cos(theta);
				vz[i] = 0;//0.1*v1;
			}
			

		}
		else{

			 		float x_vel_original = -1* rotation * v2 * sin(theta);
					float y_vel_original = 1 * rotation * v2 * cos(theta);
					float z_vel_original = 0;

					// Rotate velocities around z-axis by 60 degrees
					float x_vel_rotatedevice_z = cos(pi/3) * x_vel_original - sin(pi/3) * y_vel_original;
					float y_vel_rotatedevice_z = sin(pi/3) * x_vel_original + cos(pi/3) * y_vel_original;
					float z_vel_rotatedevice_z = z_vel_original;

					// Rotate velocities around x-axis by 30 degrees
					vx[i] = x_vel_rotatedevice_z;
					vy[i] = cos(pi/6) * y_vel_rotatedevice_z - sin(pi/6) * z_vel_rotatedevice_z;
					vz[i] = sin(pi/6) * y_vel_rotatedevice_z + cos(pi/6) * z_vel_rotatedevice_z;
		}

		// set acceleration to zero
		ax[i] = 0.0;
		ay[i] = 0.0;
		az[i] = 0.0;
	}	
		

}


void NBodiesSimulation::spatialCube(float* x, float* y, float* z, float* vx, float* vy, float* vz, float* ax, float*ay, float*az, float* mass, float p_count)
{
  	float cube_side = 20.0;  

	for (int i = 0; i < p_count; i++){
		mass[i] = 1.0;
		vx[i] = 0.0;
		vy[i] = 0.0;
		vz[i] = 0.0;
		ax[i] = 0.0;
		ay[i] = 0.0;
		az[i] = 0.0;
	}

    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, cube_side);

    int n_per_side = cbrt(p_count);

    while (n_per_side * n_per_side * n_per_side > p_count) {
        n_per_side--;
    }

    int total_gridevice_particles = n_per_side * n_per_side * n_per_side;
    int remaining_particles = p_count - total_gridevice_particles;

    float spacing = cube_side / (n_per_side - 1);
	cout<< spacing;
    int index = 0;
    for (int i = 0; i < n_per_side; ++i) {
        for (int j = 0; j < n_per_side; ++j) {
            for (int k = 0; k < n_per_side; ++k) {
                x[index] = i * spacing;
                y[index] = j * spacing;
                z[index] = k * spacing;
                ++index;
            }
        }
    }

    for (int i = 0; i < remaining_particles; ++i) {
        x[index] = distribution(generator);
        y[index] = distribution(generator);
        z[index] = distribution(generator);
        ++index;
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
