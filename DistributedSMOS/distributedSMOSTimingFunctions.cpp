/* Based on timingFunctions.cu */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <limits>

// include other files
#include "bucketMultiselect.cuh"
#include "iterativeSMOS.cuh"
#include "distributedSMOS.hpp"
#include "distributedSMOSTimingFunctions_Kernel.cuh"

// include header file
#include "distributedSMOSTimingFunctions.hpp"

#define MAX_THREADS_PER_BLOCK 1024

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {    \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)

#define RANK_NUM 4
/*
template <typename T>
struct results_t {
    float time;
    T * vals;
};
*/

/// ***********************************************************
/// ***********************************************************
/// **** MPI Message
/// ***********************************************************
/// ***********************************************************
/*

	1: each slot send h_vec to host in timeSortAndChooseMultiselect
	2: host send startSignal to each rank in timeSortAndChooseMultiselect
	3: host send stopSignal to each rank in timeSortAndChooseMultiselect
	
	8: host send startSignal to each rank in timeBucketMultiSelect
	9: host send stop to each rank in timeBucketMultiSelect
	
	22: host send startSignal to each rank in timeDistributedSMOS
	23: host send stopSignal to each rank in timeDistributedSMOS

*/

/// ***********************************************************
/// ***********************************************************
/// **** MPI Function Libraries
/// ***********************************************************
/// ***********************************************************

// TODO potential need to include this in header file

template <typename T>
void MPI_Send_CALL(T *buf, int count, 
				  int dest, int tag, MPI_Comm comm) {
	if (std::is_same<T, int>::value) {
		MPI_Send(buf, count, MPI_INT, dest, tag, comm);
	}
	else if (std::is_same<T, unsigned int>::value) {
		MPI_Send(buf, count, MPI_UNSIGNED, dest, tag, comm);
	}
	else if (std::is_same<T, float>::value) {
		MPI_Send(buf, count, MPI_FLOAT, dest, tag, comm);
	}
	else if (std::is_same<T, double>::value) {
		MPI_Send(buf, count, MPI_DOUBLE, dest, tag, comm);
	}
}

template void MPI_Send_CALL
		(int *buf, int count, int dest, int tag, MPI_Comm comm);
template void MPI_Send_CALL
		(unsigned int *buf, int count, int dest, int tag, MPI_Comm comm);
template void MPI_Send_CALL
		(float *buf, int count, int dest, int tag, MPI_Comm comm);
template void MPI_Send_CALL
		(double *buf, int count, int dest, int tag, MPI_Comm comm);

template <typename T>
void MPI_Recv_CALL(T *buf, int count, int source, int tag,
              MPI_Comm comm, MPI_Status *status) {
	if (std::is_same<T, int>::value) {
		MPI_Recv(buf, count, MPI_INT, source, tag, comm, status);
	}
	else if (std::is_same<T, unsigned int>::value) {
		MPI_Recv(buf, count, MPI_UNSIGNED, source, tag, comm, status);
	}
	else if (std::is_same<T, float>::value) {
		MPI_Recv(buf, count, MPI_FLOAT, source, tag, comm, status);
	}
	else if (std::is_same<T, double>::value) {
		MPI_Recv(buf, count, MPI_DOUBLE, source, tag, comm, status);
	}
}

template void MPI_Recv_CALL
		(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
template void MPI_Recv_CALL
		(unsigned int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
template void MPI_Recv_CALL
		(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
template void MPI_Recv_CALL
		(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
		
		


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

template<typename T>
results_t<T>* timeSortAndChooseMultiselect
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    int startSignal = 1;
    int stopSignal = 0;

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
    
    cudaThreadSynchronize();

    cudaEventRecord(start, 0);
    
    // variables for host rank
    T* h_vec_host;
    T* d_vec_host;
    T* h_vec_recv;
    
    if (rank == 0) {
    	h_vec_host = (T*)malloc(sizeof(T) * numElements * RANK_NUM);
    	cudaMalloc(&d_vec_host, sizeof(T) * numElements * RANK_NUM);
    	h_vec_recv = (T*)malloc(sizeof(T) * numElements);
    }
    
    // all rank copy vector to CPU and send it to host
    if (true) {
    	cudaMemcpy(d_vec, h_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank != 0) {
    	MPI_Send_CALL(h_vec, numElements, 0, 1, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
    	for (int j = 0; j < numElements; j++) {
    		h_vec_host[j] = h_vec[j];
    	}
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Recv_CALL(h_vec_recv, numElements, i, 1, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			for (int j = 0; j < numElements; j++) {
				h_vec_host[i * numElements + j] = h_vec_recv[j];
			}
    	}
    	
    	cudaMemcpy(d_vec_host, h_vec_host, numElements * sizeof(T) * RANK_NUM, 
    			   cudaMemcpyHostToDevice);
    	cudaThreadSynchronize();
    }
    
    // use start signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 2, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    // sorting part
	T * d_output;
	uint * d_kList;

    if (rank == 0) {
		
		sort_CALL<T>(d_vec_host, numElements * RANK_NUM);
		
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

		copyInChunk_CALL<T>(d_output, d_vec_host, d_kList, kCount, numElements * RANK_NUM, blocks, threads);
		
		cudaThreadSynchronize();
		cudaMemcpy (result->vals, d_output, kCount * sizeof (T), cudaMemcpyDeviceToHost);
		//••••••••••••••••••••••
		//printf("first result: %u \n", result->vals);
		
		cudaThreadSynchronize();

		cudaFree(d_output);
		cudaFree(d_kList);
    }
    
    // use stop signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&stopSignal, 1, i, 3, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&stopSignal, 1, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaThreadSynchronize();

    wrapupForTiming(start, stop, time, result);
    
    cudaFree(d_vec);
    if (rank == 0) {
    	free(h_vec_host);
    	free(h_vec_recv);
    	cudaFree(d_vec_host);
    }
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



// FUNCTION TO TIME Bucket Multi Select SMOS
template<typename T>
results_t<T>* timeBucketMultiSelect (T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
	T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    cudaDeviceProp dp;
    int startSignal = 1;
    int stopSignal = 0;
    
    cudaGetDeviceProperties(&dp, 0);
    
    cudaThreadSynchronize();

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

    cudaEventRecord(start, 0);
    
    // variables for host rank
    T* h_vec_host;
    T* d_vec_host;
    T* h_vec_recv;
    
    if (rank == 0) {
    	h_vec_host = (T*)malloc(sizeof(T) * numElements * RANK_NUM);
    	cudaMalloc(&d_vec_host, sizeof(T) * numElements * RANK_NUM);
    	h_vec_recv = (T*)malloc(sizeof(T) * numElements);
    }
    
    // all rank copy vector to CPU and send it to host
    if (true) {
    	cudaMemcpy(d_vec, h_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank != 0) {
    	MPI_Send_CALL(h_vec, numElements, 0, 1, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
    	for (int j = 0; j < numElements; j++) {
    		h_vec_host[j] = h_vec[j];
    	}
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Recv_CALL(h_vec_recv, numElements, i, 1, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			for (int j = 0; j < numElements; j++) {
				h_vec_host[i * numElements + j] = h_vec_recv[j];
			}
    	}
    	
    	cudaMemcpy(d_vec_host, h_vec_host, numElements * sizeof(T) * RANK_NUM, 
    			   cudaMemcpyHostToDevice);
    	cudaThreadSynchronize();
    }
    
    // use start signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 8, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {

    // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)                                        
    BucketMultiselect::bucketMultiselectWrapper(d_vec_host, numElements * RANK_NUM, kVals, kCount, result->vals, 
    											dp.multiProcessorCount, dp.maxThreadsPerBlock);
                                        
    }
                                        
    cudaThreadSynchronize();
    
    // use stop signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&stopSignal, 1, i, 9, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&stopSignal, 1, 0, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    if (rank == 0) {
    	free(h_vec_host);
    	free(h_vec_recv);
    	cudaFree(d_vec_host);
    }
    cudaThreadSynchronize();
    
    return result;
}

template results_t<int>* timeBucketMultiSelect
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeBucketMultiSelect
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeBucketMultiSelect 
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeBucketMultiSelect 
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);




// FUNCTION TO TIME ITERATIVE SMOS
template<typename T>
results_t<T>* timeIterativeSMOS (T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
	T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    cudaDeviceProp dp;
    int startSignal = 1;
    int stopSignal = 0;
    
    cudaGetDeviceProperties(&dp, 0);
    
    cudaThreadSynchronize();

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

    cudaEventRecord(start, 0);
    
    // variables for host rank
    T* h_vec_host;
    T* d_vec_host;
    T* h_vec_recv;
    
    if (rank == 0) {
    	h_vec_host = (T*)malloc(sizeof(T) * numElements * RANK_NUM);
    	cudaMalloc(&d_vec_host, sizeof(T) * numElements * RANK_NUM);
    	h_vec_recv = (T*)malloc(sizeof(T) * numElements);
    }
    
    // all rank copy vector to CPU and send it to host
    if (true) {
    	cudaMemcpy(d_vec, h_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank != 0) {
    	MPI_Send_CALL(h_vec, numElements, 0, 1, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
    	for (int j = 0; j < numElements; j++) {
    		h_vec_host[j] = h_vec[j];
    	}
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Recv_CALL(h_vec_recv, numElements, i, 1, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			for (int j = 0; j < numElements; j++) {
				h_vec_host[i * numElements + j] = h_vec_recv[j];
			}
    	}
    	
    	cudaMemcpy(d_vec_host, h_vec_host, numElements * sizeof(T) * RANK_NUM, 
    			   cudaMemcpyHostToDevice);
    	cudaThreadSynchronize();
    }
    
    // use start signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 12, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {

    // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
    IterativeSMOS::iterativeSMOSWrapper(d_vec_host, numElements * RANK_NUM, kVals, kCount, result->vals,
                                        dp.multiProcessorCount, dp.maxThreadsPerBlock);
                                        
    }
                                        
    cudaThreadSynchronize();
    
    // use stop signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&stopSignal, 1, i, 13, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&stopSignal, 1, 0, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    if (rank == 0) {
    	free(h_vec_host);
    	free(h_vec_recv);
    	cudaFree(d_vec_host);
    }
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
    int startSignal = 1;
    int stopSignal = 0;

    cudaGetDeviceProperties(&dp, 0);
    
    cudaThreadSynchronize();

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);

    cudaEventRecord(start, 0);
    
    // use start signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 22, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
    DistributedSMOS::distributedSMOSWrapper(d_vec, numElements, kVals, kCount, result->vals,
                                            dp.multiProcessorCount, dp.maxThreadsPerBlock, rank);

	cudaThreadSynchronize();
	
	// use stop signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&stopSignal, 1, i, 23, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&stopSignal, 1, 0, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
	
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



