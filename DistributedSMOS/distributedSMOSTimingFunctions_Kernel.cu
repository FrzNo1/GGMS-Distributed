#include <stdio.h>
#include <stdlib.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <limits>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include "distributedSMOSTimingFunctions_Kernel.cuh"

/// ***********************************************************
/// ***********************************************************
/// **** Thrust Functions Library
/// ***********************************************************
/// ***********************************************************

// thurst::sort function
template <typename T>
void sort_CALL(T* d_vec, unsigned int length) {
	thrust::device_ptr<T> dev_ptr(d_vec);
	thrust::sort(dev_ptr, dev_ptr + length);

}

template void sort_CALL(int* d_vec, unsigned int length);
template void sort_CALL(unsigned int* d_vec, unsigned int length);
template void sort_CALL(float* d_vec, unsigned int length);
template void sort_CALL(double* d_vec, unsigned int length);



/// ***********************************************************
/// ***********************************************************
/// **** HELPER GPU FUNCTIONS-KERNELS
/// ***********************************************************
/// ***********************************************************

template <typename T>
__global__ void copyInChunk(T * outputVector, T * inputVector, uint * kList, 
				 uint kListCount, uint numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < kListCount)
        outputVector[idx] = inputVector[numElements - kList[idx]];
        
    __syncthreads();

}



/// ***********************************************************
/// ***********************************************************
/// **** HELPER GPU FUNCTIONS LIBRARIES
/// ***********************************************************
/// ***********************************************************

template <typename T>
void copyInChunk_CALL(T * outputVector, T * inputVector, uint * kList, 
					  uint kListCount, uint numElements, int blocks, 
					  int threads) {
					  
	copyInChunk<T><<<blocks, threads>>>
				(outputVector, inputVector, kList, kListCount, numElements);

}

template void copyInChunk_CALL
			(int * outputVector, int * inputVector, uint * kList, 
		     uint kListCount, uint numElements, int blocks, 
		     int threads);
template void copyInChunk_CALL
			(unsigned int * outputVector, unsigned int * inputVector, 
			 uint * kList, uint kListCount, uint numElements, int blocks, 
		     int threads);
template void copyInChunk_CALL
			(float * outputVector, float * inputVector, uint * kList, 
		     uint kListCount, uint numElements, int blocks, 
		     int threads);
template void copyInChunk_CALL
			(double * outputVector, double * inputVector, uint * kList, 
		     uint kListCount, uint numElements, int blocks, 
		     int threads);
					  
					  
					  



