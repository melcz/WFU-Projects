#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

//-------- Generation of matrix of random single point precision numbers --------//
float * generateMatrix(int n){
  float* matrix = (float *)malloc(n*n*sizeof(float));
  for(int i=0; i<n*n; i++){
    matrix[i] = (float)(rand()%60) + drand48();
  }
  return matrix;
}

//-------- Save matrix to dat file --------//
void saveMatrixToFile(float *matrix, int size, const char *mode){
  FILE *filePointer = fopen( "product.dat", mode);

  for (int i = 0; i < size*size; i++) {
    fprintf(filePointer, "%.6g\t", matrix[i]);
    if (i%size == size-1 && i != 0) {
      fprintf(filePointer, "\n");
    }
  }

  fprintf(filePointer, "\n-------------------------------------\n");
  fclose(filePointer);
}

//-------- Matrix multiplication --------//
__global__ void multiplyMatrices(float *matrix1, float *matrix2, float *resultMatrix, int size){
  float sum = 0;
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y;

  if (i < size && j < size) {
    for (int k = 0; k < size; k++) {
      sum += matrix1[(size*i)+k]*matrix2[(size*k)+j];
    }
  }
  resultMatrix[(i*size)+j]=sum;
}

int main (int argc, char *argv[]) {
//--Testing parameters
  if (argc != 2){
      printf("Incorrect number of parameters :(\n");
      printf("Try: \"./MatrixMult <MATRIX SIZE>\"\n");
      exit(0);
  }
  int size = atoi(argv[1]);
  int tileSize = 16;

//--Generating matrices
  float *matrix1, *matrix2, *resultMatrix;
  float *dev_matrix1, *dev_matrix2, *dev_resultMatrix;

  int memorySize = size*size*sizeof(float);
  srand48(time(NULL));
  matrix1 = generateMatrix(size);
  matrix2 = generateMatrix(size);
  resultMatrix = (float *)malloc(memorySize);

//--Initializing CUDA memory
  cudaMalloc((void **)&dev_matrix1, memorySize);
  cudaMalloc((void **)&dev_matrix2, memorySize);
  cudaMalloc((void **)&dev_resultMatrix, memorySize);

  cudaMemcpy(dev_matrix1, matrix1, memorySize, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_matrix2, matrix2, memorySize, cudaMemcpyHostToDevice);

  //-- Multiplying matrices
  dim3 dimBlock(tileSize, tileSize);
  dim3 dimGrid((int)ceil((float)size/(float)dimBlock.x), (int)ceil((float)size/(float)dimBlock.y));

  clock_t start = clock();
  multiplyMatrices<<<dimGrid, dimBlock>>>(dev_matrix1, dev_matrix2, dev_resultMatrix, size);
  cudaThreadSynchronize();
  clock_t stop = clock();
  double time = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Execution time: %f seconds\n", time);

  cudaMemcpy(resultMatrix, dev_resultMatrix, memorySize, cudaMemcpyDeviceToHost);
  cudaFree(dev_matrix1); cudaFree(dev_matrix2); cudaFree(dev_resultMatrix);

//-- Saving matrices to file
  saveMatrixToFile(matrix1, size, "w");
  saveMatrixToFile(matrix2, size, "a");
  saveMatrixToFile(resultMatrix, size, "a");

  free(matrix1); free(matrix2); free(resultMatrix);
  exit(0);
}
