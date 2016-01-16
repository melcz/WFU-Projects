#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define BLOCK_SIZE 768

#define N 9999    // number of bodies
#define MASS 0     // row in array for mass
#define X_POS 1    // row in array for x position
#define Y_POS 2    // row in array for y position
#define Z_POS 3    // row in array for z position
#define X_VEL 4    // row in array for x velocity
#define Y_VEL 5    // row in array for y velocity
#define Z_VEL 6    // row in array for z velocity
#define G 10       // "gravitational constant" (not really)
#define MU 0.001   // "frictional coefficient"
#define BOXL 100.0 // periodic boundary box length

#define F_X 0
#define F_Y 1
#define F_z 2

void printStates(float **bodyStates, int tmax) {
  for (int step = 0; step <= tmax; step++) {
    // print out initial positions in PDB format
    printf("MODEL %8d\n", 0);
    for (int i = 0; i < N; i++) {
      printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
             "ATOM", i+1, "CA ", "GLY", "A", i+1,
             bodyStates[step][i*7+X_POS],
             bodyStates[step][i*7+Y_POS],
             bodyStates[step][i*7+Z_POS], 1.00, 0.00);
    }
    printf("TER\nENDMDL\n");
  }
}

//-------- Random initialization --------//
__global__ void initRand(unsigned int seed, curandState_t *states) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int index = j*blockDim.x + i;
	if (index >= N)
		return;
  curand_init(seed, index, 0, &states[index]);
}

//-------- Initialize Bodies --------//
__global__ void initializeBodies(curandState_t *states, float *bodies){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int index = j*blockDim.x + i;
	if (index >= N)
		return;
  int k = index*7;
  bodies[k+MASS] = 0.001;
  bodies[k+X_POS] = curand_uniform(&states[index]);
  bodies[k+Y_POS] = curand_uniform(&states[index]);
  bodies[k+Z_POS] = curand_uniform(&states[index]);
  bodies[k+X_VEL] = curand_uniform(&states[index]);
  bodies[k+Y_VEL] = curand_uniform(&states[index]);
  bodies[k+Z_VEL] = curand_uniform(&states[index]);
}

//-------- Update nBody timestep --------//
__global__ void calculateNBodyTimestep(curandState_t *states, float *bodiesIn, float *bodiesOut){
	float dt = 0.05; // time interval
	int k = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int x = j*blockDim.x + k;

	if (x >= N)
		return;

  float Fx_dir = 0; 
	float Fy_dir = 0; 
	float Fz_dir = 0;

  for (int i = 0; i < N; i++) {   // all other bodies
		// position differences in x-, y-, and z-directions
		float x_diff, y_diff, z_diff;

		if (i != x) {

	  	//calculate position difference between body i and x in x-,y-, and z-directions
    	x_diff = bodiesIn[i*7 + X_POS] - bodiesIn[x*7 + X_POS];
    	y_diff = bodiesIn[i*7 + Y_POS] - bodiesIn[x*7 + Y_POS];
    	z_diff = bodiesIn[i*7 + Z_POS] - bodiesIn[x*7 + Z_POS];

      // periodic boundary conditions
	  	if (x_diff <  -BOXL * 0.5) x_diff += BOXL;
	  	if (x_diff >=  BOXL * 0.5) x_diff -= BOXL;
	  	if (y_diff <  -BOXL * 0.5) y_diff += BOXL;
	  	if (y_diff >=  BOXL * 0.5) y_diff -= BOXL;
	  	if (z_diff <  -BOXL * 0.5) z_diff += BOXL;
	  	if (z_diff >=  BOXL * 0.5) z_diff -= BOXL;

	  	// calculate distance (r)
	  	float rr = (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
	  	float r = sqrt(rr);

	  	// force between bodies i and x
	  	float F = 0;

	  	// if sufficiently far away, gravitation force
	  	if (r > 2.0) {
	    	// compute gravitational force between body i and x
      	float Fg = (float)(bodiesIn[i*7 + MASS] * bodiesIn[x*7 + MASS] * G) / rr;
				
	    	// compute frictional force
				float randomFriction = curand_uniform(&states[x]) - 0.5;
      	float frictional = MU * randomFriction;
      	F = Fg + frictional;

	    	Fx_dir += (F * x_diff) / r;  // resolve forces in x and y directions
	    	Fy_dir += (F * y_diff) / r;  // and accumulate forces
	    	Fz_dir += (F * z_diff) / r;  //

	  	} else {
	    	// if too close, weak anti-gravitational force
	    	float F = G * 0.01 * 0.01 / r;
	    	Fx_dir -= F * x_diff / r;  // resolve forces in x and y directions
	    	Fy_dir -= F * y_diff / r;  // and accumulate forces
	    	Fz_dir -= F * z_diff / r;  //
			}
    }
  }

	bodiesOut[x*7 + MASS] = bodiesIn[x*7 + MASS];

  // update velocities
  bodiesOut[x*7 + X_VEL] = bodiesIn[x*7 + X_VEL] + (Fx_dir*dt)/bodiesIn[x*7 + MASS];
  bodiesOut[x*7 + Y_VEL] = bodiesIn[x*7 + Y_VEL] + (Fy_dir*dt)/bodiesIn[x*7 + MASS];
  bodiesOut[x*7 + Z_VEL] = bodiesIn[x*7 + Z_VEL] + (Fz_dir*dt)/bodiesIn[x*7 + MASS];

  // periodic boundary conditions
  if (bodiesOut[x*7 + X_VEL] <  -BOXL * 0.5) bodiesOut[x*7 + X_VEL] += BOXL;
  if (bodiesOut[x*7 + X_VEL] >=  BOXL * 0.5) bodiesOut[x*7 + X_VEL] -= BOXL;
  if (bodiesOut[x*7 + Y_VEL] <  -BOXL * 0.5) bodiesOut[x*7 + Y_VEL] += BOXL;
  if (bodiesOut[x*7 + Y_VEL] >=  BOXL * 0.5) bodiesOut[x*7 + Y_VEL] -= BOXL;
  if (bodiesOut[x*7 + Z_VEL] <  -BOXL * 0.5) bodiesOut[x*7 + Z_VEL] += BOXL;
  if (bodiesOut[x*7 + Z_VEL] >=  BOXL * 0.5) bodiesOut[x*7 + Z_VEL] -= BOXL;

  // update positions
  bodiesOut[x*7 + X_POS] = bodiesIn[x*7 + X_POS] + bodiesOut[x*7 + X_VEL]*dt;
  bodiesOut[x*7 + Y_POS] = bodiesIn[x*7 + Y_POS] + bodiesOut[x*7 + Y_VEL]*dt;
  bodiesOut[x*7 + Z_POS] = bodiesIn[x*7 + Z_POS] + bodiesOut[x*7 + Z_VEL]*dt;

  // periodic boundary conditions
  if (bodiesOut[x*7 + X_POS] <  -BOXL * 0.5) bodiesOut[x*7 + X_POS] += BOXL;
  if (bodiesOut[x*7 + X_POS] >=  BOXL * 0.5) bodiesOut[x*7 + X_POS] -= BOXL;
  if (bodiesOut[x*7 + Y_POS] <  -BOXL * 0.5) bodiesOut[x*7 + Y_POS] += BOXL;
  if (bodiesOut[x*7 + Y_POS] >=  BOXL * 0.5) bodiesOut[x*7 + Y_POS] -= BOXL;
  if (bodiesOut[x*7 + Z_POS] <  -BOXL * 0.5) bodiesOut[x*7 + Z_POS] += BOXL;
  if (bodiesOut[x*7 + Z_POS] >=  BOXL * 0.5) bodiesOut[x*7 + Z_POS] -= BOXL;
}

//-------- Main --------//
int main (int argc, char *argv[]) {
//--Testing parameters
  if (argc != 2){
      printf("Incorrect number of parameters :(\n");
      printf("Try: \"./nbody <TIME STEPS>\"\n");
      exit(0);
  }

  int tmax = atoi(argv[1]);

  if(tmax < 0){
      printf("Negative parameter not allowed.  :P\n");
      printf("Try: \"./nbody <TIME STEPS>\"\n");
      exit(0);
  }

//--Initializing variables
  float **nBodyStates, *dev_bodiesIn, *dev_bodiesOut;
  int bodiesMemorySize = N*7*sizeof(float);
  
  nBodyStates = (float **) malloc((tmax+1)*sizeof(float *));
		for(int i=0;i<=tmax;i++)
    	nBodyStates[i]=(float *) malloc(bodiesMemorySize);

//--Initializing CUDA memory
  cudaSetDevice(1);
  cudaMalloc((void **)&dev_bodiesIn, bodiesMemorySize);
	cudaMalloc((void **)&dev_bodiesOut, bodiesMemorySize);

  int blockNumber = ceil((float)N/BLOCK_SIZE);

//--Initializing Random States
  curandState_t *states;
  cudaMalloc((void**) &states, N*sizeof(curandState_t));
  initRand<<<blockNumber, BLOCK_SIZE>>>(time(NULL), states);
  cudaThreadSynchronize();

//--Initialize bodies
  initializeBodies<<<blockNumber, BLOCK_SIZE>>>(states, dev_bodiesIn);
  cudaThreadSynchronize();
	cudaMemcpy(nBodyStates[0], dev_bodiesIn, bodiesMemorySize, cudaMemcpyDeviceToHost);

//--Compute nbody changes for t
  for (int t = 1; t <= tmax; t++) {
		cudaMemcpy(dev_bodiesIn, nBodyStates[t-1], bodiesMemorySize, cudaMemcpyHostToDevice);
    calculateNBodyTimestep<<<blockNumber, BLOCK_SIZE>>>(states, dev_bodiesIn, dev_bodiesOut);
    cudaThreadSynchronize();
		cudaMemcpy(nBodyStates[t], dev_bodiesOut, bodiesMemorySize, cudaMemcpyDeviceToHost);
  }
  
  cudaFree(dev_bodiesIn); cudaFree(dev_bodiesOut); cudaFree(states);
  printStates(nBodyStates, tmax);
  free(nBodyStates);
  exit(0);
}
