SHELL = /bin/sh
CUDA_INSTALL_PATH ?= /usr/local/cuda

CPP := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc -ccbin /usr/bin
.SUFFIXES: .c .cpp .cu .o

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I./lib/ 
# Libraries
#LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcurand -lm -lgsl -lgslcblas
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcurand -lm 
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

PROGRAMS = \
compareAlgorithms \
compareMultiselect \
analyzeMultiselect \
realDataTests \
compareTopkselect 

SMOS = \
SMOStimingsOSDistrAll \
SMOStimingsOSDistrUniform \
SMOStimingsVectorGrowth \
SMOStimingsTableData \
SMOSanalyze 

CompareAlgorithms = \
compareAlgorithms.cu \
bucketSelect.cu randomizedBucketSelect.cu noExtremaRandomizedBucketSelect.cu \
generateProblems.cu timingFunctions.cu

CompareMultiselect = \
compareMultiselect.cu \
bucketMultiselect.cu naiveBucketMultiselect.cu \
generateProblems.cu multiselectTimingFunctions.cu

CompareTopkselect = \
compareTopkselect.cu \
randomizedTopkSelect.cu \
generateProblems.cu multiselectTimingFunctions.cu

AnalyzeMultiselect = \
analyzeMultiselect.cu \
bucketMultiselect.cu \
multiselectTimingFunctions.cu

RealDataTests = \
realDataTests.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOStimingsOSDistrAll = \
SMOStimingsOSDistrAll.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOStimingsOSDistrUniform = \
SMOStimingsOSDistrUniform.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOStimingsVectorGrowth = \
SMOStimingsVectorGrowth.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOStimingsTableData = \
SMOStimingsTableData.cu \
bucketMultiselect.cu \
generateProblems.cu 

SMOSanalyze = \
SMOSanalyze.cu \
bucketMultiselect.cu \
generateProblems.cu 


all: $(PROGRAMS)

allSMOS: $(SMOS)

compareAlgorithms: $(CompareAlgorithms)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

compareMultiselect: $(CompareMultiselect)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

compareTopkselect: $(CompareTopkselect)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

analyzeMultiselect: $(AnalyzeMultiselect)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

realDataTests: $(RealDataTests)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

SMOStimingsOSDistrAll: $(SMOStimingsOSDistrAll)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

SMOStimingsVectorGrowth: $(SMOStimingsVectorGrowth)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

SMOStimingsOSDistrUniform: $(SMOStimingsOSDistrUniform)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

SMOStimingsTableData: $(SMOStimingsTableData)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

SMOSanalyze: $(SMOSanalyze)
	$(NVCC) -o $@ $(addsuffix .cu,$@) $(NVCCFLAGS)

ProcessData: readMultiselectOutputfile.cpp
	$(CXX) -o readMultiselectOutput readMultiselectOutputfile.cpp $(CXXFLAGS)
 
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(PROGRAMS) *~ *.o

cleanSMOS:
	rm -rf $(SMOS) *~ *.o

#compareAlgorithms: compareAlgorithms.cu bucketSelect.cu randomizedBucketSelect.cu
#	$(NVCC) -o compareAlgorithms compareAlgorithms.cu $(NVCCFLAGS)
