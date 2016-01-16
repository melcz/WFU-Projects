#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 768

//-------- Save values to dat file --------//
void saveFrequenciesToFile(int *array, int size){
  FILE *filePointer = fopen("freq.dat", "w");

  for (int i = 0; i < 10; i++) {
    fprintf(filePointer, "0.%d, %f\n", i, (float)array[i]/(size*2));
  }
}

////-------- Random initialization --------//
__global__ void initRand(unsigned int seed, curandState_t *states) {
  curand_init(seed, threadIdx.x, 0, &states[threadIdx.x]);
}

//-------- Calculate random --------//
__global__ void calculateRandomNumbers(curandState_t *states, int *frequencies, int *result, int size){
  __shared__ int partialCount;

  if (threadIdx.x == 0) {
    partialCount = 0.0;
  }
  __syncthreads();

  if (threadIdx.x < size) {
    float rand1 = curand_uniform(&states[threadIdx.x]);
    float rand2 = curand_uniform(&states[threadIdx.x]);

    if ((rand1*rand1) + (rand2*rand2) <= 1) {
      atomicAdd(&partialCount, 1);
    }
    atomicAdd(&frequencies[(int)(rand1 * 10)], 1);
    atomicAdd(&frequencies[(int)(rand2 * 10)], 1);

    __syncthreads();

    if (threadIdx.x == 0) {
      atomicAdd(&result[0], partialCount);
    }
    __syncthreads();
  }
}

int main (int argc, char *argv[]) {
  //-------- Testing parameters --------//
    if (argc != 2){
        printf("Incorrect number of parameters :(\n");
        printf("Try: \"./MatrixMult <MATRIX SIZE>\"\n");
        exit(0);
    }

    int size = atoi(argv[1]);

    if(size < 0){
        printf("Negative parameter not allowed.  :P\n");
        printf("Try: \"./MatrixMult <MATRIX SIZE>\"\n");
        exit(0);
    }

//--Initializing variables
  int *frequencies, *dev_frequencies;
  int *result, *dev_result;

  int memorySize = 10*sizeof(int);
  srand48(time(NULL));
  frequencies = (int *)malloc(memorySize);
  result = (int *)malloc(sizeof(int));
  result[0] = 0.0;

//--Initializing CUDA memory
  cudaMalloc((void **)&dev_frequencies, memorySize);
  cudaMalloc((void **)&dev_result, sizeof(int));

  cudaMemcpy(dev_frequencies, frequencies, memorySize, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_result, result, sizeof(int), cudaMemcpyHostToDevice);

  int blockNumber = ceil((float)size/BLOCK_SIZE);

//--Initializing Random States
  curandState_t *states;
  cudaMalloc((void**) &states, size*sizeof(curandState_t));

  initRand<<<blockNumber, BLOCK_SIZE>>>(time(NULL), states);

  //--Calculate Pi
  calculateRandomNumbers<<<blockNumber, BLOCK_SIZE>>>(states, dev_frequencies, dev_result, size);
  cudaThreadSynchronize();

  cudaMemcpy(frequencies, dev_frequencies, memorySize, cudaMemcpyDeviceToHost);
  cudaMemcpy(result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

  float pi = (((float)result[0] / (float)size)*4);
  printf("Pi approximated value is: %f\n", pi);

  cudaFree(frequencies); cudaFree(dev_result);

//-- Saving matrices to file
  saveFrequenciesToFile(frequencies, size);
  free(frequencies); free(result);
  exit(0);
}
