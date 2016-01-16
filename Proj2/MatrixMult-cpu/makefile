
CC = gcc -std=c99
CFLAGS  = -g -Wall

default: MatrixMult

MatrixMult:  MatrixMult.o
	$(CC) $(CFLAGS) -o MatrixMult MatrixMult.o

MatrixMult.o:  MatrixMult.c
	$(CC) $(CFLAGS) -c MatrixMult.c

clean:
	$(RM) MatrixMult *.o *~ product.dat
