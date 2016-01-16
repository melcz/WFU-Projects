#include <stdio.h>

__global__ void decode (char *originalMessage, char *decodedMessage);

int main (int argc, char *argv[]) {

//-------- Testing parameters --------//
  if (argc != 2){
      printf("Incorrect number of parameters :(\n");
      printf("Try: \"./DecodeEmail2FULP <filename>\"\n");
      exit(0);
  }

//-------- Reading file --------//
  FILE *inputFile = fopen(argv[1], "r");

  if (inputFile == NULL) {
    fprintf(stderr, "File could not be opened :P\n");
    printf("Make sure you spelled the name of your file correctly.\n");
    exit(0);
  }

//-------- Calculating size of file and buffers--------//
  fseek(inputFile, 0, SEEK_END);
  int messageSize = ftell(inputFile);
  fseek(inputFile, 0, SEEK_SET);
  messageSize++;

  int memorySize = messageSize * sizeof(char);
  char message[messageSize], decodedMessage[messageSize];
  char *dev_message, *dev_decodedMsg;

//-------- Reading file into buffer --------//
//File is expected to be a single line without change line characters.
  while(fgets(message, messageSize, inputFile)) {
    printf("%s\n", message);
  }
  message[messageSize-1] = '\0';
  fclose(inputFile);

  printf("Decoding original message:\n %s\n", message);

//-------- Executing CUDA code --------//
  cudaMalloc((void**)&dev_message, memorySize);
  cudaMalloc((void**)&dev_decodedMsg, memorySize);

  cudaMemcpy(dev_message, message, memorySize, cudaMemcpyHostToDevice);
  decode<<<1,messageSize>>>(dev_message, dev_decodedMsg);
  cudaThreadSynchronize();
  cudaMemcpy(decodedMessage, dev_decodedMsg, memorySize, cudaMemcpyDeviceToHost);

  decodedMessage[messageSize-1] = '\0';
  printf("Decoded message is: \n%s\n",decodedMessage);

  cudaFree(dev_message);
  cudaFree(dev_decodedMsg);

  exit(0);
}

__global__ void decode (char *originalMessage, char *decodedMessage) {
// Super secret and complicated decryption algorithm.
  int i = threadIdx.x;
  decodedMessage[i] = originalMessage[i] - 1;
}
