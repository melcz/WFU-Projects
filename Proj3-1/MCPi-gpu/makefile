MCPi : $(ALL)

MCPi.o : MCPi.cu

MCPi: MCPi.o
	nvcc -o MCPi MCPi.cu

clean:
	rm -rf *.o MCPi freq.dat
