#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#define BLOCK_SIZE 32768
#define STREAMS_IN_GPU 4

__global__ void countDigits (char *digits, int *count, int maxIndex);
__global__ void initCount (int *count);

//-------- Save values to dat file --------//
void saveFrequenciesToFile(int** array0, int** array1, int total){
  FILE *filePointer = fopen("freq.dat", "w");
    for(int k = 0; k < 10; k++) {
			int sum = 0;
      for(int i = 0; i < STREAMS_IN_GPU; i++) {
				sum += array0[i][k] + array1[i][k];
      }
			fprintf(filePointer, "%d, %f\n", k, ((float)sum/(float)total));
    }
	fclose(filePointer);
}

void printDigits(char *digits, int size){
	for(int i = 0; i < size; i++){
		if ('0' <= digits[i] && digits[i] <= '9')
			printf("%c", digits[i]);
	}
	printf("\n");
}

//-------- Count method --------//
int countInGPU(cudaStream_t* streams, int fileLength, int fileRead, char* digits, int **dev_count, char **dev_digits, FILE *inputFile, int **counts) {
  int sizeToRead[STREAMS_IN_GPU];
  for(int i = 0; i < STREAMS_IN_GPU; i++) {
		sizeToRead[i] = (fileLength - (fileRead + BLOCK_SIZE)) < 0 ? fileLength - fileRead : BLOCK_SIZE;
    fread(&digits[fileRead], sizeToRead[i], 1, inputFile);
    cudaMemcpyAsync(dev_digits[i], digits+fileRead, sizeToRead[i]*sizeof(char), cudaMemcpyHostToDevice, streams[i]);
    fileRead += sizeToRead[i];
  }
  for(int i = 0; i < STREAMS_IN_GPU; i++) {
    countDigits<<<sizeToRead[i], 1, 0, streams[i]>>>(dev_digits[i], dev_count[i], sizeToRead[i]);
  }
  for(int i = 0; i < STREAMS_IN_GPU; i++) {
    cudaMemcpyAsync(counts[i], dev_count[i], 10*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
  }
  return fileRead;
}

//-------- Main --------//
int main(int argc, char **argv) {

  //--- Testing parameters
  if (argc != 2){
      printf("Incorrect number of parameters :(\n");
      printf("Try: \"./Multi-Pi <filename>\"\n");
      exit(0);
  }

//--- Reading file
  FILE *inputFile = fopen(argv[1], "r");

  if (inputFile == NULL) {
    fprintf(stderr, "File could not be opened :P\n");
    printf("Make sure you spelled the name of your file correctly.\n");
    exit(0);
  }

//--- Calculating size of file
  fseek(inputFile, 0, SEEK_END);
  int fileLength = ftell(inputFile);
  fseek(inputFile, 0, SEEK_SET);
	int memorySize = fileLength * sizeof(char);
	printf("File length:%d memorySize:%d\n", fileLength, memorySize);

//--- Declaring CUDA and host variables
	cudaStream_t* streams0 = (cudaStream_t*) malloc(STREAMS_IN_GPU*sizeof(cudaStream_t));
  cudaStream_t* streams1 = (cudaStream_t*) malloc(STREAMS_IN_GPU*sizeof(cudaStream_t));
  int **counts0 = (int**) malloc(STREAMS_IN_GPU*sizeof(int*));
	for(int i = 0; i < STREAMS_IN_GPU; i++) {
    counts0[i] = (int*) malloc(sizeof(int)*10);
	}
  int **counts1 = (int**) malloc(STREAMS_IN_GPU*sizeof(int*));
	for(int i = 0; i < STREAMS_IN_GPU; i++) {
    counts1[i] = (int*) malloc(sizeof(int)*10);
	}
  int **dev_count0 = (int **)malloc(STREAMS_IN_GPU*sizeof(int*));
  int **dev_count1 = (int **)malloc(STREAMS_IN_GPU*sizeof(int*));
	char **dev_digits0 = (char **)malloc(STREAMS_IN_GPU*sizeof(char*));
  char **dev_digits1 = (char **)malloc(STREAMS_IN_GPU*sizeof(char*));
	char *digits0, *digits1;
	
	//--- Initializing GPU 0
	cudaSetDevice(0);
  for(int j = 0; j < STREAMS_IN_GPU; j++) {
    cudaStreamCreate(&streams0[j]);
    cudaMalloc((void**)&dev_digits0[j], BLOCK_SIZE*sizeof(char));
    cudaMalloc((void**)&dev_count0[j], 10*sizeof(int));
    initCount<<<10, 1, 0, streams0[j]>>>(dev_count0[j]);
  }

	//--- Initializing GPU 1
	cudaSetDevice(1);
  for(int j = 0; j < STREAMS_IN_GPU; j++) {
    cudaStreamCreate(&streams1[j]);
    cudaMalloc((void**)&dev_digits1[j], BLOCK_SIZE*sizeof(char));
    cudaMalloc((void**)&dev_count1[j], 10*sizeof(int));
    initCount<<<10, 1, 0, streams1[j]>>>(dev_count1[j]);
  }

	//--- Executing CUDA code
  clock_t start = clock();

	cudaSetDevice(0);
  for(int j = 0; j < STREAMS_IN_GPU; j++) {
    initCount<<<10, 1, 0, streams0[j]>>>(dev_count0[j]);
  }
	cudaSetDevice(1);
  for(int j = 0; j < STREAMS_IN_GPU; j++) {
    initCount<<<10, 1, 0, streams1[j]>>>(dev_count1[j]);
  }

	//--- Initializing GPU 1
	cudaSetDevice(1);
  for(int j = 0; j < STREAMS_IN_GPU; j++) {
    cudaStreamCreate(&streams1[j]);
    cudaMalloc((void**)&dev_digits1[j], BLOCK_SIZE*sizeof(char));
    cudaMalloc((void**)&dev_count1[j], 10*sizeof(int));
    initCount<<<10, 1, 0, streams1[j]>>>(dev_count1[j]);
  }

	cudaSetDevice(0);
  cudaDeviceSynchronize();
	cudaHostAlloc((void **)&digits0, fileLength*sizeof(char), cudaHostAllocDefault);
	cudaSetDevice(1);
  cudaDeviceSynchronize();
  cudaHostAlloc((void **)&digits1, fileLength*sizeof(char), cudaHostAllocDefault);

  int GPU = 0;
	int i = 0;

  while(i < fileLength) {
    GPU %= 2;
		cudaSetDevice(GPU);
    if(GPU == 0) {
      i = countInGPU(streams0, fileLength, i, digits0, dev_count0, dev_digits0, inputFile, counts0);
    } else {
      i = countInGPU(streams1, fileLength, i, digits1, dev_count1, dev_digits1, inputFile, counts1);
    }
    GPU++;
	}

  cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaSetDevice(1);
  cudaDeviceSynchronize();


  printf("Devices were synchronized\n");
	fclose(inputFile);
  saveFrequenciesToFile(counts0, counts1, fileLength);

  cudaSetDevice(0);
  for(int i = 0; i < STREAMS_IN_GPU; i++) {
		cudaFreeHost(digits0);
    cudaFree(dev_digits0[i]);
    cudaFree(dev_count0[i]);
    cudaStreamDestroy(streams0[i]);
  }
  cudaFree(digits0);

  cudaSetDevice(1);
	for(int i = 0; i < STREAMS_IN_GPU; i++) {
		cudaFreeHost(digits1);
    cudaFree(dev_digits1[i]);
    cudaFree(dev_count1[i]);
    cudaStreamDestroy(streams1[i]);
  }
  cudaFree(digits1);

	clock_t stop = clock();
  double time = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Operation time:%f\n", time);

  exit(0);
}

__global__ void initCount (int *count) {
  int index = blockIdx.x;
  count[index]=0;
}

__global__ void countDigits (char *digits, int *count, int maxIndex) {
  int index = blockIdx.x;
	int number = digits[index] - '0'; 
  if(number >= 0 && number < 10 && index < maxIndex){
      atomicAdd(&count[number], 1);
  }
}

