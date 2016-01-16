
CC = gcc -std=c99
CFLAGS  = -g -Wall

default: lawrence

lawrence:  lawrence.o list.o problem.o calculations.o
	$(CC) $(CFLAGS) -o lawrence lawrence.o list.o problem.o calculations.o

lawrence.o:  lawrence.c
	$(CC) $(CFLAGS) -c lawrence.c

list.o : list.c list.h
	$(CC) $(CFLAGS) -c list.c

problem.o : problem.c problem.h
	$(CC) $(CFLAGS) -c problem.c

calculations.o : calculations.c calculations.h
	$(CC) $(CFLAGS) -c calculations.c

clean:
	$(RM) lawrence *.o *~
