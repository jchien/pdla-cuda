# Makefile for cuda PDLA

CC = gcc
NVCC = nvcc

CCOPTS = -O3 -Wall -I/u/local/cuda/current/include/
NVCCOPTS = -arch=sm_20 -I/u/local/cuda/current/SDK/C/common/inc/

all: pdla

pdla: cuda.o main.o
	$(CC) $(CCOPTS) -lm \
		-L/u/local/cuda/current/lib64/ -L/u/local/cuda/current/SDK/C/lib/ -lcudart -lcutil_x86_64 \
		-o pdla cuda.o main.o
		
%.o : %.cpp
	$(CC) $(CCOPTS) -c $< -o $@ 
	
%.o : %.cu
	$(NVCC) $(NVCCOPTS) -c $< -o $@ 
	
clean:
	rm -f pdla *.o