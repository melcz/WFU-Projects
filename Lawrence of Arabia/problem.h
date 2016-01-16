#include <stdio.h>
#include <stdlib.h>

typedef struct Problem
{
	int bombs;
	int depots;
	int *values;
  int ***minimumValues;
  int ***bombDistribution;
} Problem;

void printProblem(void* data);
void printMatrix(int** matrix, int size);
void freeProblem(void* data);
void printBombPositions(Problem *problem);
