#include "calculations.h"

void calculateStrategicValues(Problem * problem) {
  for(int i=0; i<problem->depots; i++)
    for(int j=0; j<problem->depots; j++)
      problem->minimumValues[0][i][j] = 0;

  int l,i,j;
  for(l=1; l<problem->depots; l++){
    for(i=0; i<problem->depots-l; i++){
      j=l+i;
      problem->minimumValues[0][i][j] =
        problem->minimumValues[0][i+1][j] +
        problem->minimumValues[0][i][j-1] -
        problem->minimumValues[0][i+1][j-1] +
        (problem->values[i]*problem->values[j]);
    }
  }
}

void lawrenceOfArabia(Problem *problem){
  int m,k,l,i,j, min, pos, temp;
  //looping through tables for optimal values with m bombs
  for(m=1; m<=problem->bombs; m++){
    for(l=0; l<problem->depots; l++){
      for(i=0; i<problem->depots-l; i++){
        j=l+i;
        min = INT_MAX;
        int maxPosition = (j-i)-m+i;
        for (k = i; k <= maxPosition; k++) {
          temp = problem->minimumValues[0][i][k]+ problem->minimumValues[m-1][k+1][j];
          if (temp < min) {
            min = temp;
            pos = k;
          }
        }
        problem->minimumValues[m][i][j] = min;
        problem->bombDistribution[m][i][j] = pos;
      }
    }
  }
}

void solveLawrenceOfArabia(Problem *problem){
  //Initializing three dimensional matrix to solve problem, indexed [subI][subJ][bombs]
  problem->minimumValues = malloc((problem->bombs+1)*sizeof(int**));
  for(int i=0; i<=problem->bombs; i++){
    problem->minimumValues[i] = malloc(problem->depots*sizeof(int*));
    for(int j=0; j<problem->depots;j++){
      problem->minimumValues[i][j] = malloc(problem->depots*sizeof(int));
    }
  }

  //Initializing three dimensional matrix to keep track of bomb positions
  problem->bombDistribution = malloc((problem->bombs+1)*sizeof(int**));
  for(int i=0; i<=problem->bombs; i++){
    problem->bombDistribution[i] = malloc(problem->depots*sizeof(int*));
    for(int j=0; j<problem->depots;j++){
      problem->bombDistribution[i][j] = malloc(problem->depots*sizeof(int));
    }
  }

  calculateStrategicValues(problem);
  lawrenceOfArabia(problem);
  printProblem(problem);
}

//Recursive algorithm
// int lawrenceArabia(int bombs, int i, int j, Problem *problem){
//   printf("Lawrence with %d bombs, from %d to %d\n", bombs, i, j);
//   int min = INT_MAX; int temp;
//   int maxPosition = (j-i)-bombs+i;
//
//   for (int k = i; k <= maxPosition; k++) {
//     printf("Iterating %d to %d and %d to %d\n", i, k, k+1, j);
//     if(bombs == 1) {
//       temp = problem->strategicValues[i][k] + problem->strategicValues[k+1][j];
//     } else {
//       temp = problem->strategicValues[i][k] + lawrenceArabia(bombs-1, k+1, j, problem);
//     }
//     printf("temp: %d\n", temp);
//     if (temp < min) {
//       min = temp;
//     }
//   }
//   printf("\n");
//   return min;
// }
