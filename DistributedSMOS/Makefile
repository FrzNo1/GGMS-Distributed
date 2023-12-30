SHELL = /bin/sh
CUDA_INSTALL_PATH ?= /usr/local/cuda

 CPP := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc -ccbin /usr/bin
MPICPP := mpic++
.SUFFIXES: .c .cpp .cu .o

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I./lib/ 
# Libraries
#LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcurand -lm -lgsl -lgslcblas
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcurand -lm 
LIB_MPICUDA := -L$(CUDA_INSTALL_PATH)/lib64
# ARCH
ARCH = -arch=sm_60

# Common flags
 COMMONFLAGS += $(INCLUDES)
# Compilers
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(ARCH)
NVCCFLAGS += $(LIB_CUDA)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
MPICPPFLAGS += $(COMMONFLAGS)
MPICPPFLAGS += $(LIB_MPICUDA)
MPICPPLINK += $(COMMONFLAGS)
MPICPPLINK += $(LIB_CUDA)
MPICPPLINK += -lcudart


ITER = \
compareIterativeSMOS

DIST = \
compareDistributedSMOS

CompareIterativeSMOSSRC = \
bucketMultiselect.cu iterativeSMOS.cu \
generateProblems.cu iterativeSMOSTimingFunctions.cu \
printFunctions.cu compareIterativeSMOS.cu

CompareIterativeSMOSOBJS = $(CompareIterativeSMOSSRC:.cu=.o)

CompareDistributedSMOSSRC = \
distributedSMOS.cpp distributedSMOS_Kernel.cu \
iterativeSMOS.cu bucketMultiselect.cu \
distributedSMOSTimingFunctions_Kernel.cu \
distributedSMOSTimingFunctions.cpp \
generateProblems.cu printFunctions.cu \
compareDistributedSMOS.cpp

CompareDistributedSMOSOBJS = $(CompareDistributedSMOSSRC:.cu=.o)
CompareDistributedSMOSOBJS := $(CompareDistributedSMOSOBJS:.cpp=.o)




iterative: $(ITER)

distributed: $(DIST)

compareIterativeSMOS: $(CompareIterativeSMOSOBJS)
	$(NVCC) $(NVCCFLAGS) $(CompareIterativeSMOSOBJS) -o $@
	
compareDistributedSMOS: $(CompareDistributedSMOSOBJS)
	$(MPICPP) $(CompareDistributedSMOSOBJS) $(MPICPPLINK) -o $@
	
ProcessData: readDistriSMOSoutputfile.cpp
	$(CXX) -o readDistriSMOSoutput readDistriSMOSoutputfile.cpp $(CXXFLAGS)
	
# folloing files needs MPICPP COMPILER	
distributedSMOS.o: distributedSMOS.cpp
	$(MPICPP) -c $(MPICPPFLAGS) $< -o $@
	
distributedSMOSTimingFunctions.o: distributedSMOSTimingFunctions.cpp
	$(MPICPP) -c $(MPICPPFLAGS) $< -o $@
	
compareDistributedSMOS.o: compareDistributedSMOS.cpp
	$(MPICPP) -c $(MPICPPFLAGS) $< -o $@
	
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@
	
.PHONY: clean
clean:
	rm -f *.o compareIterativeSMOS compareDistributedSMOS readDistriSMOSoutput
