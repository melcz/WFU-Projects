Multi-Pi : $(ALL)

Multi-Pi.o : Multi-Pi.cu

Multi-Pi: Multi-Pi.o
	nvcc -o Multi-Pi Multi-Pi.cu

clean:
	rm -rf *.o freq.dat Multi-Pi
