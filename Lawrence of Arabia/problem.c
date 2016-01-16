#include "problem.h"

void freeProblem(void* data){
  Problem *problem = data;
  free(problem->values);
  for(int i=0; i<=problem->bombs; i++){
    for(int j=0; j<problem->depots;j++){
      free(problem->minimumValues[i][j]);
    }
    free(problem->minimumValues[i]);
  }
  free(problem->minimumValues);

  for(int i=0; i<=problem->bombs; i++){
    for(int j=0; j<problem->depots;j++){
      free(problem->bombDistribution[i][j]);
    }
    free(problem->bombDistribution[i]);
  }
  free(problem->bombDistribution);
  free(problem);
}

void printProblem(void* data)
{
  Problem *problem = data;
  printf("___________________________________________________\n");
  printf("Bombs: %d Depots: %d\n", problem->bombs, problem->depots);
  for (int i = 0; i < problem->depots; i++) {
    printf("%d ", problem->values[i]);
  }
  printf("\nStarting strategic value: %d\n", problem->minimumValues[0][0][problem->depots-1]);
  printf("Minimal strategic value: %d\n", problem->minimumValues[problem->bombs][0][problem->depots-1]);
  printf("Bomb positions:\n");
  printBombPositions(problem);
  printf("\n___________________________________________________\n\n");
}

void printBombPositions(Problem *problem) {
  int m,k,i,j;
  i = 0;
  j = problem->depots-1;
  for (m = problem->bombs; m > 0; m--) {
    k=problem->bombDistribution[m][i][j];
    i=k+1;
    printf("%d ", k);
  }
}

void printMatrix(int** matrix, int size) {
  for(int i=0; i<size; i++) {
    for(int j=0; j<size; j++){
      printf("%d ", matrix[i][j]);
    }
    printf("\n");
  }
}
