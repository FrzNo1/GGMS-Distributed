SHELL = /bash/sh
CUDA_INSTALL_PATH ?= /usr/local/cuda

CC = gcc
NVCC = nvcc -ccbin /usr/bin

# Includes
INCLUDES = -I./lib/ -I.
# Libraries

# ARCH
ARCH = -arch=sm_60


# Add /usr/local/cuda/bin to the PATH variable
export PATH := $(PATH):/usr/local/cuda/bin



CompareIterativeSMOS: iterativeSMOS.o bucketMultiselect.o iterativeSMOSTimingFunctions.o generateProblems.o printFunctions.o compareIterativeSMOS.o

bucketMultiselect.o

iterativeSMOS.o: 
	 

.PHONY: clean
clean:
	rm -f *.o compareIterativeSMOS compareDistributedSMOS
