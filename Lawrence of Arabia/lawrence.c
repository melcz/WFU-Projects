#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "list.h"
#include "calculations.h"

void readProblems(list* problemList) {
  while (true) {
    int depots, bombs;
    if (scanf("%d %d", &depots, &bombs)){

      if (depots == 0)
        break;
      Problem *problem = malloc(sizeof(Problem));
      problem->depots = depots; problem->bombs = bombs;

      problem->values = malloc(depots*sizeof(int));
      for (int i = 0; i < depots; i++) {
        scanf("%d", &problem->values[i]);
      }

      push_back(problemList, problem);

    } else {
      int ch;
      while ((ch=getchar()) != EOF && ch != '\n');
      printf("Invalid input line, try again\n");
    }
  }
}

int main()
{
  list* problemList = create_list();
  printf("Awaiting input: \n");

  readProblems(problemList);

  for (int i = 0; i < problemList->size; i++) {
    solveLawrenceOfArabia(get_index(problemList, i));
  }

  empty_list(problemList, freeProblem);
  printf("\n\n");
  return 0;
}
