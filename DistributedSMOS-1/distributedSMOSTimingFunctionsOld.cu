/* Based on timingFunctions.cu */
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
#include <cub/cub.cuh>

// include other files
#include "bucketMultiselect.cuh"
#include "iterativeSMOS.cuh"
#include "distributedSMOS.hpp"

// include header file
#include "distributedSMOSTimingFunctionsOld.cuh"

#define MAX_THREADS_PER_BLOCK 1024

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {    \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)

/*
template <typename T>
struct results_t {
    float time;
    T * vals;
};
*/


template <typename T>
void setupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T * h_vec, T ** d_vec,
                    results_t<T> ** result, uint numElements, uint kCount) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(d_vec, numElements * sizeof(T));
    cudaMemcpy(*d_vec, h_vec, numElements * sizeof(T), cudaMemcpyHostToDevice);

    *result = (results_t<T> *) malloc (sizeof (results_t<T>));
    (*result)->vals = (T *) malloc (kCount * sizeof (T));
    
    cudaThreadSynchronize();
}

template <typename T>
void wrapupForTiming(cudaEvent_t &start, cudaEvent_t &stop, float time, results_t<T> * result) {
    result->time = time;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaThreadSynchronize();
}

/////////////////////////////////////////////////////////////////
//          THE SORT AND CHOOSE TIMING FUNCTION
/////////////////////////////////////////////////////////////////


template <typename T>
__global__ void copyInChunk(T * outputVector, T * inputVector, uint * kList, uint kListCount, uint numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < kListCount)
        outputVector[idx] = inputVector[numElements - kList[idx]];

}

template<typename T>
results_t<T>* timeSortAndChooseMultiselect
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
    
    cudaThreadSynchronize();

    cudaEventRecord(start, 0);
    thrust::device_ptr<T> dev_ptr(d_vec);
    thrust::sort(dev_ptr, dev_ptr + numElements);
    
    cudaThreadSynchronize();

    /*
    for (int i = 0; i < kCount; i++)
      cudaMemcpy(result->vals + i, d_vec + (numElements - kVals[i]), sizeof (T), cudaMemcpyDeviceToHost);
    */

    T * d_output;
    uint * d_kList;

    cudaMalloc (&d_output, kCount * sizeof (T));
    cudaMalloc (&d_kList, kCount * sizeof(uint));
    cudaMemcpy (d_kList, kVals, kCount * sizeof (uint), cudaMemcpyHostToDevice);
    
    cudaThreadSynchronize();

    int threads = MAX_THREADS_PER_BLOCK;
    if (kCount < threads)
        threads = kCount;
    int blocks = (int) ceil (kCount / (float) threads);
    
    cudaThreadSynchronize();

    copyInChunk<T><<<blocks, threads>>>(d_output, d_vec, d_kList, kCount, numElements);
    
    cudaThreadSynchronize();
    cudaMemcpy (result->vals, d_output, kCount * sizeof (T), cudaMemcpyDeviceToHost);
//••••••••••••••••••••••
    //printf("first result: %u \n", result->vals);

    cudaFree(d_output);
    cudaFree(d_kList);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaThreadSynchronize();

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    
    cudaThreadSynchronize();
    
    return result;
}

template results_t<int>* timeSortAndChooseMultiselect
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeSortAndChooseMultiselect
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeSortAndChooseMultiselect
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeSortAndChooseMultiselect
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);



// FUNCTION TO TIME BUCKET MULTISELECT
template<typename T>
results_t<T>* timeBucketMultiselect (T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
    
    cudaThreadSynchronize();

    cudaEventRecord(start, 0);

    // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
    BucketMultiselect::bucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result->vals,
                                                dp.multiProcessorCount, dp.maxThreadsPerBlock);
                                                
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    
    cudaThreadSynchronize();
    
    return result;
}

template results_t<int>* timeBucketMultiselect 
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeBucketMultiselect 
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeBucketMultiselect 
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeBucketMultiselect 
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);


// FUNCTION TO TIME NAIVE BUCKET MULTISELECT
template<typename T>
results_t<T>* timeNaiveBucketMultiselect (T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

    cudaEventRecord(start, 0);
    thrust::device_ptr<T> dev_ptr(d_vec);
    thrust::sort(dev_ptr, dev_ptr + numElements);

    for (int i = 0; i < kCount; i++)
        cudaMemcpy(result->vals + i, d_vec + (numElements - kVals[i]), sizeof (T), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    
    cudaThreadSynchronize();
    
    return result;
}

template results_t<int>* timeNaiveBucketMultiselect 
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeNaiveBucketMultiselect 
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeNaiveBucketMultiselect 
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeNaiveBucketMultiselect 
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);



// FUNCTION TO TIME ITERATIVE SMOS
template<typename T>
results_t<T>* timeIterativeSMOS (T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
    
    cudaThreadSynchronize();

    cudaEventRecord(start, 0);

    // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
    IterativeSMOS::iterativeSMOSWrapper(d_vec, numElements, kVals, kCount, result->vals,
                                        dp.multiProcessorCount, dp.maxThreadsPerBlock);
                                        
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    
    cudaThreadSynchronize();
    
    return result;
}

template results_t<int>* timeIterativeSMOS 
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeIterativeSMOS 
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeIterativeSMOS 
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeIterativeSMOS 
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);


// FUNCTION TO TIME DISTRIBUTED SMOS
template<typename T>
results_t<T>* timeDistributedSMOS (T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    cudaDeviceProp dp;

    cudaGetDeviceProperties(&dp, 0);
    
    cudaThreadSynchronize();

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

    cudaEventRecord(start, 0);

    // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
    DistributedSMOS::distributedSMOSWrapper(d_vec, numElements, kVals, kCount, result->vals,
                                            dp.multiProcessorCount, dp.maxThreadsPerBlock, rank);

	cudaThreadSynchronize();
	
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    
    cudaThreadSynchronize();
    
    return result;
}

template results_t<int>* timeDistributedSMOS 
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeDistributedSMOS 
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeDistributedSMOS 
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeDistributedSMOS 
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);





/***************************************
********* TOP K SELECT
****************************************/

template<typename T>
results_t<T>* timeSortAndChooseTopkselect(T * h_vec, uint numElements, uint kCount) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

    cudaEventRecord(start, 0);
    thrust::device_ptr<T> dev_ptr(d_vec);
    thrust::sort(dev_ptr, dev_ptr + numElements, thrust::greater<T>());

    cudaMemcpy(result->vals, d_vec, kCount * sizeof(T), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    return result;
}

template results_t<int>* timeSortAndChooseTopkselect
	(int * h_vec, uint numElements, uint kCount);
template results_t<unsigned int>* timeSortAndChooseTopkselect
	(unsigned int * h_vec, uint numElements, uint kCount);
template results_t<float>* timeSortAndChooseTopkselect
	(float * h_vec, uint numElements, uint kCount);
template results_t<double>* timeSortAndChooseTopkselect
	(double * h_vec, uint numElements, uint kCount);

/*
// FUNCTION TO TIME RANDOMIZED TOP K SELECT
template<typename T>
results_t<T>* timeRandomizedTopkselect (T * h_vec, uint numElements, uint kCount) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

    cudaEventRecord(start, 0);
    result->vals = randomizedTopkSelectWrapper(d_vec, numElements, kCount);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    return result;
}

template results_t<int>* timeRandomizedTopkselect 
	(int * h_vec, uint numElements, uint kCount);
template results_t<unsigned int>* timeRandomizedTopkselect 
	(unsigned int * h_vec, uint numElements, uint kCount);
template results_t<float>* timeRandomizedTopkselect 
	(float * h_vec, uint numElements, uint kCount);
template results_t<double>* timeRandomizedTopkselect 
	(double * h_vec, uint numElements, uint kCount);
*/

// FUNCTION TO TIME BUCKET TOP K SELECT
template<typename T>
results_t<T>* timeBucketTopkselect (T * h_vec, uint numElements, uint kCount) {
    // initialize ks
    uint * kVals = (uint *) malloc(kCount*sizeof(T));
    for (uint i = 0; i < kCount; i++)
        kVals[i] = i+1;

    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

    cudaEventRecord(start, 0);

    BucketMultiselect::bucketMultiselectWrapper(d_vec, numElements, kVals, kCount, result->vals, dp.multiProcessorCount, dp.maxThreadsPerBlock);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    return result;
}

template results_t<int>* timeBucketTopkselect 
	(int * h_vec, uint numElements, uint kCount);
template results_t<unsigned int>* timeBucketTopkselect 
	(unsigned int * h_vec, uint numElements, uint kCount);
template results_t<float>* timeBucketTopkselect 
	(float * h_vec, uint numElements, uint kCount);
template results_t<double>* timeBucketTopkselect 
	(double * h_vec, uint numElements, uint kCount);
