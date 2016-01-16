#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

//-------- Save values to dat file --------//
void saveFrequenciesToFile(int *array, int size){
  FILE *filePointer = fopen("freq.dat", "w");

  for (int i = 0; i < 10; i++) {
    fprintf(filePointer, "0.%d, %.6g\n", i, (float)array[i]/(float)(size*2));
  }
}

float calculatePi(int size, int *counts){
  float inCircle = 0.0;
  for (int i = 0; i < 2*size; i+=2) {
    float rand1 = (float)drand48();
    float rand2 = (float)drand48();
    if ((rand1*rand1) + (rand2*rand2) <= 1) {
      inCircle++;
    }
    counts[(int)(rand1 * 10)]++;
    counts[(int)(rand2 * 10)]++;
  }
  float result = ((inCircle / size)*4);
  return result;
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

  //-------- Init random --------//
  srand48(time(NULL));

//-------- Pi calculation --------//
  int *frequencies = malloc(10*sizeof(int));
  for (int i = 0; i < 10; i++) {
    frequencies[i]=0;
  }

  float pi = calculatePi(size, frequencies);
  printf("Pi approximated value is: %f\n", pi);

//-------- Saving frequencies to file --------//
  saveFrequenciesToFile(frequencies, size);

  free(frequencies);
}
