nbody : $(ALL)

nbody.o : nbody.cu

nbody: nbody.o
	nvcc -o nbody nbody.cu

clean:
	rm -rf *.o nbody
