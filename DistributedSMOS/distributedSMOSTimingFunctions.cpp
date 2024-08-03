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
#include "distributedBucketMultiselect.hpp"
#include "iterativeSMOS.cuh"
#include "distributedSMOS.hpp"
#include "distributedSMOSTimingFunctions_Kernel.cuh"

// include header file
#include "distributedSMOSTimingFunctions.hpp"

#define MAX_THREADS_PER_BLOCK 1024

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {    \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)

// #define RANK_NUM 4

extern int RANK_NUM;
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

	1: rank 1 send vector to rank 0
	2: rank 3 send vector to rank 1
	2: host send startSignal to each rank in timeSortAndChooseMultiselect
	3: host send stopSignal to each rank in timeSortAndChooseMultiselect
	
	8: host send startSignal to each rank in timeBucketMultiSelect
	9: host send stop to each rank in timeBucketMultiSelect
	
	TODO: miss documentation
	11:
	12:
	13:
	14:
	
	21:
	22:
	23:
	24:
	
	31:
	32:
	33
	
	22: host send startSignal to each rank in timeDistributedSMOS
	23: host send stopSignal to each rank in timeDistributedSMOS
	24: host send startSignal to each rank in timeDistributedBucketMultiselect
	25: host send stopSignal to each rank in timeDistributedBucketMultiselect

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
void mergeSort_helper(T* vec1, T* vec2, T* mergedVec, int length) {
	int index1 = 0;
	int index2 = 0;
	int index = 0;
	
	while (index1 < length && index2 < length) {
		if (vec1[index1] < vec2[index2]) {
			mergedVec[index] = vec1[index1];
			index1++;
			index++;
		}
		else {
			mergedVec[index] = vec2[index2];
			index2++;
			index++;
		}
	
	}
	
	if (index1 < length) {
		for (int i = index1; i < length; i++) {
			mergedVec[index] = vec1[i];
			index++;
		}
	}
	else {
		for (int i = index2; i < length; i++) {
			mergedVec[index] = vec2[i];
			index++;
			
		}
	}
}	

template void mergeSort_helper(int* vec1, int* vec2, int* mergedVec, int length);
template void mergeSort_helper(unsigned int* vec1, unsigned int* vec2, 
							   unsigned int* mergedVec, int length);
template void mergeSort_helper(float* vec1, float* vec2, float* mergedVec, int length);
template void mergeSort_helper(double* vec1, double* vec2, double* mergedVec, int length);

template <typename T>
void mergeSort_helper_odd(T* vec1, T* vec2, T* mergedVec, int length1, int length2) {
	int index1 = 0;
	int index2 = 0;
	int index = 0;
	
	while (index1 < length1 && index2 < length2) {
		if (vec1[index1] < vec2[index2]) {
			mergedVec[index] = vec1[index1];
			index1++;
			index++;
		}
		else {
			mergedVec[index] = vec2[index2];
			index2++;
			index++;
		}
	
	}
	
	if (index1 < length1) {
		for (int i = index1; i < length1; i++) {
			mergedVec[index] = vec1[i];
			index++;
		}
	}
	else {
		for (int i = index2; i < length2; i++) {
			mergedVec[index] = vec2[i];
			index++;
			
		}
	}
}	

template void mergeSort_helper_odd(int* vec1, int* vec2, int* mergedVec, int length1, int length2);
template void mergeSort_helper_odd(unsigned int* vec1, unsigned int* vec2, 
							   unsigned int* mergedVec, int length1, int length2);
template void mergeSort_helper_odd(float* vec1, float* vec2, float* mergedVec, int length1, int length2);
template void mergeSort_helper_odd(double* vec1, double* vec2, double* mergedVec, int length1, int length2);


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

// test part
template<typename T>
results_t<T>* timeSortAndChooseMultiselect_original
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
		
		/*
		// test part
		printf("%d\n", kVals[341]);
		printf("%d\n", numElements - kVals[341]);
		*/
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

template results_t<int>* timeSortAndChooseMultiselect_original
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeSortAndChooseMultiselect_original
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeSortAndChooseMultiselect_original
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeSortAndChooseMultiselect_original
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
	
template<typename T>
results_t<T>* timeSortAndChooseMultiselect
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
	
	if (RANK_NUM == 2) {
		return timeSortAndChooseMultiselect_TwoRank
				(h_vec, numElements, kVals, kCount, rank);
	}
	else if (RANK_NUM == 4) {
		return timeSortAndChooseMultiselect_FourRank
				(h_vec, numElements, kVals, kCount, rank);
	}
	else if (RANK_NUM == 6) {
		return timeSortAndChooseMultiselect_SixRank
				(h_vec, numElements, kVals, kCount, rank);
	}
	else if (RANK_NUM == 8) {
		return timeSortAndChooseMultiselect_EightRank
				(h_vec, numElements, kVals, kCount, rank);
	}
	
	
	return NULL;
}

template results_t<int>* timeSortAndChooseMultiselect
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeSortAndChooseMultiselect
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeSortAndChooseMultiselect
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeSortAndChooseMultiselect
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
	
	

template<typename T>
results_t<T>* timeSortAndChooseMultiselect_TwoRank
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    int startSignal = 1;
    int stopSignal = 0;
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 201, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
    
    cudaThreadSynchronize();

    cudaEventRecord(start, 0);
    
    // variables for host rank
    T* d_vec_host;
    
    // all rank sort in GPU and copy it to CPU
    sort_CALL<T>(d_vec, numElements);
    cudaMemcpy(h_vec, d_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
   	    // use start signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 6, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // rank 1 send data to rank 0 and rank 0 performs mergesort
    T* h_vec_host_recv;
    T* h_vec_host;
    if (rank == 0) {
    	h_vec_host_recv = (T*)malloc(numElements * sizeof(T));
    	h_vec_host = (T*)malloc(numElements * 2 * sizeof(T));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 1) {
    	MPI_Send_CALL(h_vec, numElements, 0, 31, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
    	MPI_Recv_CALL(h_vec_host_recv, numElements, 1, 31, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
		mergeSort_helper(h_vec, h_vec_host_recv, h_vec_host, numElements);
		
		/*
		// test part
		for (int i = 0; i < 1000; i++) {
			if (h_vec_host[i] > h_vec_host[i + 1]) {
				printf("wrong %d   ", i);
			}
		}
		*/
    	
    }
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 203, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 203, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    // sorting part
	T * d_output;
	uint * d_kList;

    if (rank == 0) {
		cudaMalloc(&d_vec_host, sizeof(T) * numElements * RANK_NUM);
		cudaMemcpy(d_vec_host, h_vec_host, numElements * sizeof(T) * RANK_NUM, 
    			   cudaMemcpyHostToDevice);
		
		cudaThreadSynchronize();


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
		cudaFree(d_vec_host);
    }
    
    
    // use stop signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&stopSignal, 1, i, 7, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&stopSignal, 1, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaThreadSynchronize();

    wrapupForTiming(start, stop, time, result);
    
    cudaFree(d_vec);
    if (rank == 0) {
    	free(h_vec_host_recv);
    	free(h_vec_host);
    }

    cudaThreadSynchronize();
    
    
    return result;
}

template results_t<int>* timeSortAndChooseMultiselect_TwoRank
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeSortAndChooseMultiselect_TwoRank
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeSortAndChooseMultiselect_TwoRank
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeSortAndChooseMultiselect_TwoRank
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);


template<typename T>
results_t<T>* timeSortAndChooseMultiselect_FourRank
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    int startSignal = 1;
    int stopSignal = 0;
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 201, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
    
    cudaThreadSynchronize();

    cudaEventRecord(start, 0);
    
    // variables for host rank
    T* d_vec_host;
    
    // all rank sort in GPU and copy it to CPU
    sort_CALL<T>(d_vec, numElements);
    cudaMemcpy(h_vec, d_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
   	    // use start signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 6, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    // rank 3 send data to rank 2, rank 1 send data to rank 0
    // and the two nodes perform merge_sort
    T* h_vec_middle;
    T* h_vec_middle_recv;
    if (rank == 0 || rank == 2) {
    	h_vec_middle = (T*)malloc(sizeof(T) * numElements * 2);
    	h_vec_middle_recv = (T*)malloc(sizeof(T) * numElements);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 3 || rank == 1) {
    	if (rank == 1) {
    		MPI_Send_CALL(h_vec, numElements, 0, 1, MPI_COMM_WORLD);
    	}
    	if (rank == 3) {
    		MPI_Send_CALL(h_vec, numElements, 2, 2, MPI_COMM_WORLD);
    	}
    }
    
    if (rank == 2 || rank == 0) {
    	if (rank == 0) {
    		MPI_Recv_CALL(h_vec_middle_recv, numElements, 1, 1, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec, h_vec_middle_recv, h_vec_middle, numElements);
    	}
    	if (rank == 2) {
    		MPI_Recv_CALL(h_vec_middle_recv, numElements, 3, 2, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec, h_vec_middle_recv, h_vec_middle, numElements);
    	}
    }
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 202, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 202, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // rank 2 send data to rank 0 and rank 0 performs mergesort
    T* h_vec_host_recv;
    T* h_vec_host;
    if (rank == 0) {
    	free(h_vec_middle_recv);
    	h_vec_host_recv = (T*)malloc(numElements * 2 * sizeof(T));
    	h_vec_host = (T*)malloc(numElements * 4 * sizeof(T));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 2) {
    	MPI_Send_CALL(h_vec_middle, numElements * 2, 0, 3, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
    	MPI_Recv_CALL(h_vec_host_recv, numElements * 2, 2, 3, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
		mergeSort_helper(h_vec_middle, h_vec_host_recv, h_vec_host, numElements * 2);
		
		/*
		// test part
		for (int i = 0; i < 1000; i++) {
			if (h_vec_host[i] > h_vec_host[i + 1]) {
				printf("wrong %d   ", i);
			}
		}
		*/
    	
    }
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 203, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 203, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    // sorting part
	T * d_output;
	uint * d_kList;

    if (rank == 0) {
		cudaMalloc(&d_vec_host, sizeof(T) * numElements * RANK_NUM);
		cudaMemcpy(d_vec_host, h_vec_host, numElements * sizeof(T) * RANK_NUM, 
    			   cudaMemcpyHostToDevice);
		
		cudaThreadSynchronize();


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
		cudaFree(d_vec_host);
    }
    
    
    // use stop signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&stopSignal, 1, i, 7, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&stopSignal, 1, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaThreadSynchronize();

    wrapupForTiming(start, stop, time, result);
    
    cudaFree(d_vec);
    if (rank == 0) {
    	free(h_vec_middle);
    	free(h_vec_host_recv);
    	free(h_vec_host);
    }
    if (rank == 2) {
    	free(h_vec_middle);
    	free(h_vec_middle_recv);
    }
    cudaThreadSynchronize();
    
    
    return result;
}

template results_t<int>* timeSortAndChooseMultiselect_FourRank
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeSortAndChooseMultiselect_FourRank
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeSortAndChooseMultiselect_FourRank
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeSortAndChooseMultiselect_FourRank
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
	
	
template<typename T>
results_t<T>* timeSortAndChooseMultiselect_SixRank
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    int startSignal = 1;
    int stopSignal = 0;
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 201, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
    
    cudaThreadSynchronize();

    cudaEventRecord(start, 0);
    
    // variables for host rank
    T* d_vec_host;
    
    // all rank sort in GPU and copy it to CPU
    sort_CALL<T>(d_vec, numElements);
    cudaMemcpy(h_vec, d_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
   	    // use start signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 6, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
	// rank 6 send data to rank 2
    // rank 5 send data to rank 1, rank 4 send data to rank 0
    // and the four nodes perform merge_sort
    T* h_vec_upper;
    T* h_vec_upper_recv;    
    if (rank < 3) {
    	h_vec_upper = (T*)malloc(sizeof(T) * numElements * 2);
    	h_vec_upper_recv = (T*)malloc(sizeof(T) * numElements);
    }    
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank >= 3) {
    	if (rank == 3) {
    		MPI_Send_CALL(h_vec, numElements, 0, 21, MPI_COMM_WORLD);
    	}
    	if (rank == 4) {
    		MPI_Send_CALL(h_vec, numElements, 1, 22, MPI_COMM_WORLD);
    	}
    	if (rank == 5) {
    		MPI_Send_CALL(h_vec, numElements, 2, 23, MPI_COMM_WORLD);
    	}
    }
    
    if (rank < 3) {
    	if (rank == 0) {
    		MPI_Recv_CALL(h_vec_upper_recv, numElements, 4, 21, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec, h_vec_upper_recv, h_vec_upper, numElements);
    	}
    	if (rank == 1) {
    		MPI_Recv_CALL(h_vec_upper_recv, numElements, 5, 22, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec, h_vec_upper_recv, h_vec_upper, numElements);
    	}
    	if (rank == 2) {
    		MPI_Recv_CALL(h_vec_upper_recv, numElements, 6, 23, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec, h_vec_upper_recv, h_vec_upper, numElements);
    	}
    
    }
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 202, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 202, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
      
    // rank 1 send data to rank 2
    // and the two nodes perform merge_sort
    T* h_vec_middle;
    T* h_vec_middle_recv;
    if (rank == 2) {
    	// free(h_vec_upper_recv);
    	h_vec_middle = (T*)malloc(sizeof(T) * numElements * 4);
    	h_vec_middle_recv = (T*)malloc(sizeof(T) * numElements * 2);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    

	if (rank == 1) {
		MPI_Send_CALL(h_vec_upper, numElements * 2, 2, 24, MPI_COMM_WORLD);
	}

	if (rank == 2) {
		MPI_Recv_CALL(h_vec_middle_recv, numElements * 2, 1, 24, MPI_COMM_WORLD, 
					 MPI_STATUS_IGNORE);
		mergeSort_helper(h_vec_upper, h_vec_middle_recv, h_vec_middle, numElements * 2);
	}

    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 203, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 203, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    // rank 2 send data to rank 0 and rank 0 performs mergesort
    T* h_vec_host_recv;
    T* h_vec_host;
    if (rank == 0) {
    	// free(h_vec_middle_recv);
    	h_vec_host_recv = (T*)malloc(numElements * 4 * sizeof(T));
    	h_vec_host = (T*)malloc(numElements * 6 * sizeof(T));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 2) {
    	MPI_Send_CALL(h_vec_middle, numElements * 4, 0, 25, MPI_COMM_WORLD);
    }
    
    // TODO: work to right here
    
    if (rank == 0) {
    	MPI_Recv_CALL(h_vec_host_recv, numElements * 4, 2, 25, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
		mergeSort_helper_odd(h_vec_middle, h_vec_host_recv, h_vec_host, 
				numElements * 2, numElements * 4);
		
		/*
		// test part
		for (int i = 0; i < 1000; i++) {
			if (h_vec_host[i] > h_vec_host[i + 1]) {
				printf("wrong %d   ", i);
			}
		}
		*/
    	
    }
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 204, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 204, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    // sorting part
	T * d_output;
	uint * d_kList;

    if (rank == 0) {
		cudaMalloc(&d_vec_host, sizeof(T) * numElements * RANK_NUM);
		cudaMemcpy(d_vec_host, h_vec_host, numElements * sizeof(T) * RANK_NUM, 
    			   cudaMemcpyHostToDevice);
		
		cudaThreadSynchronize();


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
		cudaFree(d_vec_host);
    }
    
    
    // use stop signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&stopSignal, 1, i, 7, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&stopSignal, 1, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaThreadSynchronize();

    wrapupForTiming(start, stop, time, result);
    
    cudaFree(d_vec);
    if (rank < 3) {
    	free(h_vec_upper);
    	free(h_vec_upper_recv);  	
    }
    if (rank == 2) {
    	free(h_vec_middle);
    	free(h_vec_middle_recv);
    }
    if (rank == 0) {
    	free(h_vec_host_recv);
    	free(h_vec_host);
    }
    cudaThreadSynchronize();
    
    
    return result;
}

template results_t<int>* timeSortAndChooseMultiselect_SixRank
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeSortAndChooseMultiselect_SixRank
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeSortAndChooseMultiselect_SixRank
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeSortAndChooseMultiselect_SixRank
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);



template<typename T>
results_t<T>* timeSortAndChooseMultiselect_EightRank
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
    T * d_vec;
    results_t<T> * result;
    float time;
    cudaEvent_t start, stop;
    int startSignal = 1;
    int stopSignal = 0;
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 201, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    setupForTiming(start, stop, h_vec, &d_vec, &result, numElements, kCount);
    
    cudaThreadSynchronize();

    cudaEventRecord(start, 0);
    
    // variables for host rank
    T* d_vec_host;
    
    // all rank sort in GPU and copy it to CPU
    sort_CALL<T>(d_vec, numElements);
    cudaMemcpy(h_vec, d_vec, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
   	    // use start signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 6, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // rank 7 send data to rank 3, rank 6 send data to rank 2
    // rank 5 send data to rank 1, rank 4 send data to rank 0
    // and the four nodes perform merge_sort
    T* h_vec_upper;
    T* h_vec_upper_recv;    
    if (rank < 4) {
    	h_vec_upper = (T*)malloc(sizeof(T) * numElements * 2);
    	h_vec_upper_recv = (T*)malloc(sizeof(T) * numElements);
    }    
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank >= 4) {
    	if (rank == 4) {
    		MPI_Send_CALL(h_vec, numElements, 0, 11, MPI_COMM_WORLD);
    	}
    	if (rank == 5) {
    		MPI_Send_CALL(h_vec, numElements, 1, 12, MPI_COMM_WORLD);
    	}
    	if (rank == 6) {
    		MPI_Send_CALL(h_vec, numElements, 2, 13, MPI_COMM_WORLD);
    	}
    	if (rank == 7) {
    		MPI_Send_CALL(h_vec, numElements, 3, 14, MPI_COMM_WORLD);
    	}
    }
    
    if (rank < 4) {
    	if (rank == 0) {
    		MPI_Recv_CALL(h_vec_upper_recv, numElements, 4, 11, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec, h_vec_upper_recv, h_vec_upper, numElements);
    	}
    	if (rank == 1) {
    		MPI_Recv_CALL(h_vec_upper_recv, numElements, 5, 12, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec, h_vec_upper_recv, h_vec_upper, numElements);
    	}
    	if (rank == 2) {
    		MPI_Recv_CALL(h_vec_upper_recv, numElements, 6, 13, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec, h_vec_upper_recv, h_vec_upper, numElements);
    	}
    	if (rank == 3) {
    		MPI_Recv_CALL(h_vec_upper_recv, numElements, 7, 14, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec, h_vec_upper_recv, h_vec_upper, numElements);
    	}
    
    }
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 202, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 202, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // TODO: work to right here
    
    
    // rank 3 send data to rank 2, rank 1 send data to rank 0
    // and the two nodes perform merge_sort
    T* h_vec_middle;
    T* h_vec_middle_recv;
    if (rank == 0 || rank == 2) {
    	// free(h_vec_upper_recv);
    	h_vec_middle = (T*)malloc(sizeof(T) * numElements * 4);
    	h_vec_middle_recv = (T*)malloc(sizeof(T) * numElements * 2);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 3 || rank == 1) {
    	if (rank == 1) {
    		MPI_Send_CALL(h_vec_upper, numElements * 2, 0, 15, MPI_COMM_WORLD);
    	}
    	if (rank == 3) {
    		MPI_Send_CALL(h_vec_upper, numElements * 2, 2, 16, MPI_COMM_WORLD);
    	}
    }
    
    if (rank == 2 || rank == 0) {
    	if (rank == 0) {
    		MPI_Recv_CALL(h_vec_middle_recv, numElements * 2, 1, 15, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec_upper, h_vec_middle_recv, h_vec_middle, numElements * 2);
    	}
    	if (rank == 2) {
    		MPI_Recv_CALL(h_vec_middle_recv, numElements * 2, 3, 16, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
			mergeSort_helper(h_vec_upper, h_vec_middle_recv, h_vec_middle, numElements * 2);
    	}
    }
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 203, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 203, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // rank 2 send data to rank 0 and rank 0 performs mergesort
    T* h_vec_host_recv;
    T* h_vec_host;
    if (rank == 0) {
    	// free(h_vec_middle_recv);
    	h_vec_host_recv = (T*)malloc(numElements * 4 * sizeof(T));
    	h_vec_host = (T*)malloc(numElements * 8 * sizeof(T));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 2) {
    	MPI_Send_CALL(h_vec_middle, numElements * 4, 0, 17, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
    	MPI_Recv_CALL(h_vec_host_recv, numElements * 4, 2, 17, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
		mergeSort_helper(h_vec_middle, h_vec_host_recv, h_vec_host, numElements * 4);
		
		/*
		// test part
		for (int i = 0; i < 1000; i++) {
			if (h_vec_host[i] > h_vec_host[i + 1]) {
				printf("wrong %d   ", i);
			}
		}
		*/
    	
    }
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 204, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 204, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    // sorting part
	T * d_output;
	uint * d_kList;

    if (rank == 0) {
		cudaMalloc(&d_vec_host, sizeof(T) * numElements * RANK_NUM);
		cudaMemcpy(d_vec_host, h_vec_host, numElements * sizeof(T) * RANK_NUM, 
    			   cudaMemcpyHostToDevice);
		
		cudaThreadSynchronize();


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
		cudaFree(d_vec_host);
    }
    
    
    // use stop signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&stopSignal, 1, i, 7, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&stopSignal, 1, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaThreadSynchronize();

    wrapupForTiming(start, stop, time, result);
    
    cudaFree(d_vec);
    if (rank < 4) {
    	free(h_vec_upper);
    	free(h_vec_upper_recv);  	
    }
    if (rank == 0 || rank == 2) {
    	free(h_vec_middle);
    	free(h_vec_middle_recv);
    }
    if (rank == 0) {
    	free(h_vec_host_recv);
    	free(h_vec_host);
    }
    cudaThreadSynchronize();
    
    
    return result;
}

template results_t<int>* timeSortAndChooseMultiselect_EightRank
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeSortAndChooseMultiselect_EightRank
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeSortAndChooseMultiselect_EightRank
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeSortAndChooseMultiselect_EightRank
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
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 204, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 204, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
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
    
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&startSignal, 1, i, 205, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 205, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

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
    
    /*
    // test part
    if (rank == 0) {
			T* vector_local = (T*)malloc(numElements * sizeof(T));
			cudaMemcpy(vector_local, d_vec, numElements * sizeof(T),
					   cudaMemcpyDeviceToHost);
			for (int i = 0; i < 200; i++) {
				printf("%u  ", vector_local[i]);
			
			}
			printf("\n");
		
	}
	*/
    
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
	
	
	
// FUNCTION TO TIME DISTRIBUTED BUCKET MULTI SELECT
template<typename T>
results_t<T>* timeDistributedBucketMultiselect 
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank) {
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
    		MPI_Send_CALL(&startSignal, 1, i, 24, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&startSignal, 1, 0, 24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    /*
    // test part
    if (rank == 3) {
			T* vector_local = (T*)malloc(numElements * sizeof(T));
			cudaMemcpy(vector_local, d_vec, numElements * sizeof(T),
					   cudaMemcpyDeviceToHost);
			for (int i = 0; i < 200; i++) {
				printf("%u  ", vector_local[i]);
			
			}
			printf("\n");
		
	}
	*/
    
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    // bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, uint kCount, T * outputs, int blocks, int threads)
    DistributedBucketMultiselect::distributedBucketMultiselectWrapper
    	(d_vec, numElements, kVals, kCount, result->vals,
         dp.multiProcessorCount, dp.maxThreadsPerBlock, rank);

	cudaThreadSynchronize();
	
	// use stop signal to make sure each rank is at the same stage
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++) {
    		MPI_Send_CALL(&stopSignal, 1, i, 25, MPI_COMM_WORLD);
    	}
    }
    
    if (rank != 0) {
    	MPI_Recv_CALL(&stopSignal, 1, 0, 25, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
	
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    wrapupForTiming(start, stop, time, result);
    cudaFree(d_vec);
    
    cudaThreadSynchronize();
    
    return result;
}

template results_t<int>* timeDistributedBucketMultiselect
	(int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<unsigned int>* timeDistributedBucketMultiselect 
	(unsigned int * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<float>* timeDistributedBucketMultiselect
	(float * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
template results_t<double>* timeDistributedBucketMultiselect 
	(double * h_vec, uint numElements, uint * kVals, uint kCount, int rank);



