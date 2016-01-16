MatrixMult : $(ALL)

MatrixMult.o : MatrixMult.cu

MatrixMult: MatrixMult.o
	nvcc -o MatrixMult MatrixMult.cu

clean:
	rm -rf *.o MatrixMult
