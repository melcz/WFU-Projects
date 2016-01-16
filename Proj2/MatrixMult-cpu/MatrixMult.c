#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

//-------- Generation of matrix of random single point precision numbers --------//
float * generateMatrix(int n){
  float* matrix = malloc(n*n*sizeof(float));
  for(int i=0; i<n*n; i++){
    matrix[i] = (float)(rand()%60) + drand48();
  }
  return matrix;
}

//-------- Matrix multiplication --------//
float * multiplyMatrices(float *matrix1, float *matrix2, int size){
  float* resultMatrix = malloc(size*size*sizeof(float));
  float temp;

  for(int i = 0; i<size; i++){
    for(int j = 0; j<size; j++){
      // Calculating entry i, j of new matrix
      temp = 0;
      for(int k = 0; k < size; k++){
        temp += matrix1[(size*i)+k]*matrix2[(size*k)+j];
      }
      resultMatrix[size*i+j]=temp;
    }
  }
  return resultMatrix;
}

//-------- Save matrix to dat file --------//
void saveMatrixToFile(float *matrix, int size, char *mode){
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

int main (int argc, char *argv[]) {
//-------- Testing parameters --------//
  if (argc != 2){
      printf("Incorrect number of parameters :(\n");
      printf("Try: \"./MatrixMult <MATRIX SIZE>\"\n");
      exit(0);
  }

//-------- Generating matrices --------//
  int size = atoi(argv[1]);
  srand48(time(NULL));
  float *matrix1 = generateMatrix(size);
  float *matrix2 = generateMatrix(size);

//-------- Matrix multiplication --------//
  clock_t start = clock();
  float* result = multiplyMatrices(matrix1, matrix2, size);
  clock_t stop = clock();
  double time = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Execution time: %f seconds\n", time);

//-------- Saving matrices to file --------//
  saveMatrixToFile(matrix1, size, "w");
  saveMatrixToFile(matrix2, size, "a");
  saveMatrixToFile(result, size, "a");

  free(matrix1); free(matrix2); free(result);
}
