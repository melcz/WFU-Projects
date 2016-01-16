
CC = gcc -std=c99
CFLAGS  = -g -Wall

default: MCPi

MatrixMult:  MCPi.o
	$(CC) $(CFLAGS) -o MCPi MCPi.o

MatrixMult.o:  MCPi.c
	$(CC) $(CFLAGS) -c MCPi.c

clean:
	$(RM) MCPi *.o *~ freq.dat
