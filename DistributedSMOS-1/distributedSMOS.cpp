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
#include "distributedSMOS_Kernel.cuh"

// include h files
#include "distributedSMOS.hpp"

#define RANK_NUM 4
#define PROBLEM_SIZE 100000
#define NUM_K_SIZE 100
			


/// ***********************************************************
/// ***********************************************************
/// **** MPI Message
/// ***********************************************************
/// ***********************************************************
/*
	0: host send array to each slot in main
	1: each slot send minimum to host in distributedSMOS, STEP 1.1
	2: each slot send maximum to host in distributedSMOS, STEP 1.1
	3: host send numUniqueBuckets to each slot in distributedSMOS, STEP 2.2
	4: host send slops to each slot in distributedSMOS, STEP 2.2
	5: host send pivotsLeft to each slot in distributedSMOS, STEP 2.2
	6: host send pivotsRight to each slot in distributedSMOS, STEP 2.2
	7: host send kthnumBuckets to each slot in distributedSMOS, STEP 2.2
	8: each slot send h_bucketCount to host in distributedSMOS, STEP 4.1
	9: host send tempKorderLength to each slot in distributedSMOS, STEP 4.3
	10: host send tempKorderBucket to each slot in distributedSMOS, STEP 4.3
	11: host send tempKorderIndeces to each slot in distributedSMOS, STEP 4.3
	12: host send numKs to each slot in distributedSMOS, STEP 4.3
	13: each slot send tempOutput to host in distributedSMOS, STEP 4.3
	14: each slot send sampleVector to host in distributedSMOS, STEP 2.1
	
	30: each slot send length_local to host in distributedSMOS, STEP 1.1
	31: host send maximum_host to each slot in distributedSMOS, STEP 1.1
	32: host send minimum_host to each slot in distributedSMOS, STEP 1.1
	
	50: host send numUniqueBuckets to each slot in distributedSMOS, STEP 5.1.1
	51: host send uniqueBuckets to each slot in distributedSMOS, STEP 5.1.1
	52:	host send slopes to each slot in distributedSMOS, STEP 5.3.2
	53: host send pivotsLeft to each slot in distributedSMOS, STEP 5.3.2
	54: host send pivotsRight to each slot in distributedSMOS, STEP 5.3.2
	55: host send kthnumBuckets to each slot in distributedSMOS, STEP 5.3.2
	56: each slot send h_bucketCount to host in distributedSMOS, STEP 5.5.1
	57: host send tempKorderLength to each slot in distributedSMOS, STEP 5.5.3
	58: host send tempKorderBucket to each slot in distributedSMOS, STEP 5.5.3
	59: host send tempKorderIndeces to each slot in distributedSMOS, STEP 5.5.3
	60: host send numKs to each slot in distributedSMOS, STEP 5.5.3
	61: host send length_host to each slot in dsitributedSMOS, STEP 5.5.3
	62: host send length_host_Old to each slot in distributedSMOS, STEP 5.5.3
	63: each slot send tempOutput to host in distributedSMOS, STEP 5.5.3
	
	64: host send kthBuckets to each slot in distributedSMOS, STEP 6
	
	1001: 

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



namespace DistributedSMOS {
	using namespace std;


	/// ***********************************************************
	/// ***********************************************************
	/// **** distributedSMOS: the main algorithm
	/// ***********************************************************
	/// ***********************************************************


	/**
	 * This function is the main process of the algorithm. 
	 * It reduces the given multi-selection problem to a smaller 
	 * problem by using bucketing ideas with multiple computers.
	*/
	template <typename T>
	T distributedSMOS (T* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
					   int numKs, T* output, int blocks, int threads, int numBuckets, int numPivots, 
					   int rank) {
/*
    template <typename T>
	T distributedSMOS (T* d_vector_local, int length_local, int rank) {
*/
		/// ***********************************************************
		/// **** STEP 1: Initialization
		/// **** STEP 1.1: Find Min and Max of the whole vector
		/// **** Find Min and Max in each slot and send it to parent
		/// **** We don't need to go through the rest of the algorithm if it's flat
		/// ***********************************************************
		
		/*
		// test part
		if (rank == 0) {
			printf("\n\n\n\n\n");
			printf("Distributed Version\n");
		}
		*/
		
		T maximum_local, minimum_local;
		T maximum_host, minimum_host;
		T maximum_each_receive, minimum_each_receive;
		
		if (true) {
			minmax_element_CALL(d_vector_local, length_local, &maximum_local, &minimum_local);

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
	    
		    if (rank != 0) {
				MPI_Send_CALL(&minimum_local, 1, 0, 1, MPI_COMM_WORLD);
				MPI_Send_CALL(&maximum_local, 1, 0, 2, MPI_COMM_WORLD);
			}

		}
		

		MPI_Barrier(MPI_COMM_WORLD);

		
		if (rank == 0) {
			// receive maximum and minimum of vector from each slots
			maximum_host = maximum_local;
			minimum_host = minimum_local;

			for (int i = 1; i < RANK_NUM; i++) {
				MPI_Recv_CALL(&minimum_each_receive, 1, i, 1, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
				MPI_Recv_CALL(&maximum_each_receive, 1, i, 2, MPI_COMM_WORLD, 
	     				 MPI_STATUS_IGNORE);
				if (maximum_each_receive > maximum_host) {
					maximum_host = maximum_each_receive;
				}
				if (minimum_each_receive < minimum_host) {
					minimum_host = minimum_each_receive;
				}
			}
			
			for (int i = 1; i < RANK_NUM; i++) {
				MPI_Send_CALL(&maximum_host, 1, i, 31, MPI_COMM_WORLD);
				MPI_Send_CALL(&minimum_host, 1, i, 32, MPI_COMM_WORLD);
			}
		}
		
		if (rank != 0) {
			MPI_Recv_CALL(&maximum_host, 1, 0, 31, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv_CALL(&minimum_host, 1, 0, 32, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		

		/*
		// test part
		printf("max: %d, min: %d\n", maximum_host, minimum_host);
		*/

			
		if (rank == 0) {
			//if the max and the min are the same, then we are done
			if (maximum_host == minimum_host) {
				for (int i = 0; i < numKs; i++)
					output[i] = minimum_host;
			}
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		if (maximum_host == minimum_host) {
			return 1;
		}


		/// ***********************************************************
		/// **** STEP 1: Initialization
		/// **** STEP 1.2: Declare variables and allocate memory
		/// **** Declare Variables
		/// ***********************************************************

			
		// declare variables for kernel to launch
		int threadsPerBlock = threads;
		int numBlocks = blocks;
		int offset = blocks * threads;

		
		// variables for the randomized selection
		int sampleSize_host;
		if (length_total < (1<<15)) {
			sampleSize_host = (1<<12);
		}
		else if (length_total < (1<<20)) {
			sampleSize_host = (1<<14);
		}
		else if (length_total < (1<<25)) {
			sampleSize_host = (1<<16);
		}
		else if (length_total < (1<<28)) {
			sampleSize_host = (1<<18);
		}
		else{
			sampleSize_host = (1<<20);
		}
	
		int sampleSize_local = sampleSize_host / RANK_NUM;
		T* sampleVector = (T*)malloc(sampleSize_local * sizeof(T));
		T* d_sampleVector;
		cudaMalloc(&d_sampleVector, sampleSize_local * sizeof(T));
		
		// pivots variables
		// potential to simplify
		int numSpaceAllocate;
		if (numKs > numPivots)
			numSpaceAllocate = numKs;
		else
			numSpaceAllocate = numPivots;

		double* slopes = (double*)malloc(numSpaceAllocate * sizeof(double));
		double* d_slopes;
		T* pivots = (T*)malloc(numPivots * sizeof(T));
		T* d_pivots;
		cudaMalloc(&d_slopes, numSpaceAllocate * sizeof(double));
		cudaMalloc(&d_pivots, numPivots * sizeof(T));
		
		T * pivotsLeft = (T*)malloc(numSpaceAllocate * sizeof(T));                                 // new variables
        T * pivotsRight = (T*)malloc(numSpaceAllocate * sizeof(T));
        T * d_pivotsLeft;
        T * d_pivotsRight;
        T * d_newPivotsLeft;
        T * d_newPivotsRight;
        cudaMalloc(&d_pivotsLeft, numSpaceAllocate * sizeof(T));
        cudaMalloc(&d_pivotsRight, numSpaceAllocate * sizeof(T));
        cudaMalloc(&d_newPivotsLeft, numSpaceAllocate * sizeof(T));
        cudaMalloc(&d_newPivotsRight, numSpaceAllocate * sizeof(T));
        
        
        // Allocate memory to store bucket assignments
        size_t size = length_local * sizeof(unsigned int);
        unsigned int* d_elementToBucket;    //array showing what bucket every element is in
        cudaMalloc(&d_elementToBucket, size);
        
        
        // Allocate memory to store bucket counts
        size_t totalBucketSize = numBlocks * numBuckets * sizeof(unsigned int);
        unsigned int * h_bucketCount = (unsigned int *) malloc (numBuckets * sizeof (unsigned int));
        //array showing the number of elements in each bucket
        unsigned int * d_bucketCount;
        cudaMalloc(&d_bucketCount, totalBucketSize);
        
		
		// Allocate memory to store the new vector for kVals
        T * d_newvector;
        cudaMalloc(&d_newvector, length_local * sizeof(T));
        T * addressOfd_newvector = d_newvector;
		

		// array of kth buckets
		int numUniqueBuckets;
		int numUniqueBucketsOld;
		int length_host = length_total;    
		int length_host_Old;
		int length_local_Old = length_local;
		int tempKorderLength;
		unsigned int* d_kVals;
		unsigned int* kthBuckets = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int));       	
		unsigned int* d_kthBuckets;                                                                      	
		unsigned int* kthBucketScanner = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int)); 	// potential to host only
		unsigned int* kIndices = (unsigned int*)malloc(numKs * sizeof(unsigned int));
		unsigned int* d_kIndices;
		unsigned int* uniqueBuckets = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int));    	
		unsigned int* d_uniqueBuckets;    																 	
		unsigned int* uniqueBucketCounts = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int)); 	// potential to host only
		unsigned int* d_uniqueBucketCounts;    																// potential to host only
		unsigned int* reindexCounter = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int));   	
		unsigned int* d_reindexCounter;
		unsigned int* kthnumBuckets = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int));
        unsigned int* d_kthnumBuckets;
        T * tempOutput = (T *)malloc(numSpaceAllocate * sizeof(T));											
        T * d_tempOutput;																					
        unsigned int * tempKorderBucket = (unsigned int *)malloc(numSpaceAllocate * sizeof(unsigned int));
        unsigned int * d_tempKorderBucket;																		
        unsigned int * tempKorderIndeces = (unsigned int *)malloc(numSpaceAllocate * sizeof(unsigned int));
        unsigned int * d_tempKorderIndeces;
		cudaMalloc(&d_kVals, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_kIndices, numKs * sizeof(unsigned int));
		cudaMalloc(&d_kthBuckets, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_uniqueBuckets, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_uniqueBucketCounts, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_reindexCounter, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_kthnumBuckets, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_tempOutput, numSpaceAllocate * sizeof(T));
        cudaMalloc(&d_tempKorderBucket, numSpaceAllocate * sizeof(unsigned int));
        cudaMalloc(&d_tempKorderIndeces, numSpaceAllocate * sizeof(unsigned int));

		for (int i = 0; i < numKs; i++) {
			kIndices[i] = i;
		}
		
		
		// arrays for host to receive information
		unsigned int * h_bucketCount_Host = NULL;
		unsigned int * h_bucketCount_Receive = NULL;
		T * tempOutput_Receive = NULL;
		T * sampleVector_Host = NULL;
		T * d_sampleVector_Host = NULL;
		
		if (rank == 0) {
			h_bucketCount_Host = (unsigned int *)malloc(numBuckets * sizeof(unsigned int));
			h_bucketCount_Receive = (unsigned int *)malloc(numBuckets * sizeof(unsigned int));
			tempOutput_Receive = (T *)malloc(numSpaceAllocate * sizeof(T));
			sampleVector_Host = (T *)malloc(sampleSize_host * sizeof(T));
			cudaMalloc(&d_sampleVector_Host, sampleSize_host * sizeof(T));
		}

		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		



		/// ***********************************************************
		/// **** STEP 1: Initialization
		/// **** STEP 1.3: Sort the klist
		/// **** and we have to keep the old index
		/// ***********************************************************

		if (true) {
			cudaMemcpy(d_kIndices, kIndices, numKs * sizeof(unsigned int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_kVals, kVals, numKs * sizeof(unsigned int), cudaMemcpyHostToDevice);

			sort_by_key_CALL(d_kVals, d_kIndices, numKs);
			
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);

		    cudaMemcpy(kIndices, d_kIndices, numKs * sizeof (unsigned int), cudaMemcpyDeviceToHost);
		    cudaMemcpy(kVals, d_kVals, numKs * sizeof (unsigned int), cudaMemcpyDeviceToHost);
		}
		
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		/*
		// test part
		if (rank == 0) {
			printf("rank 0, finishing STEP 1\n");
		}
		if (rank == 2) {
			printf("rank 2, finishing STEP 1\n");
		}
		*/
		
		
		
		
		
		/// ***********************************************************
        /// **** STEP 2: CreateBuckets
        /// **** STEP 2.1: Collect sample
        /// **** Collect samples from each rank
        /// ***********************************************************
        
        
        // randomly select numbers from the all ranks
        if (true) {
		    generateSamples_distributive_CALL
				(d_vector_local, d_sampleVector, length_local, sampleSize_local, offset);
				
			cudaMemcpy(sampleVector, d_sampleVector, sampleSize_local * sizeof(T), 
					   cudaMemcpyDeviceToHost);
					   
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
	    
		    if (rank != 0) {
				MPI_Send_CALL(sampleVector, sampleSize_local, 0, 14, MPI_COMM_WORLD);
			}
		}
		
		if (rank == 0) {
			for (int i = 0; i < sampleSize_local; i++) {
				sampleVector_Host[i] = sampleVector[i];
			}
			
			for (int i = 1; i < RANK_NUM; i++) {
				MPI_Recv_CALL(sampleVector, sampleSize_local, i, 14, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
					     
				for (int j = 0; j < sampleSize_local; j++) {
					sampleVector_Host[i * sampleSize_local + j] = sampleVector[j];
				}
			}
			
			cudaMemcpy(d_sampleVector_Host, sampleVector_Host, sampleSize_host * sizeof(T), 
					   cudaMemcpyHostToDevice);
		}
		
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);


        
        /// ***********************************************************
        /// **** STEP 2: CreateBuckets
        /// **** STEP 2.2: Declare and Generate Pivots and Slopes
        /// **** Host generate KD intervals and send it to other ranks
        /// ***********************************************************
        
        // Find bucket sizes using a randomized selection
        if (rank == 0) {
		    generatePivots<T>(pivots, slopes, d_sampleVector_Host, sampleSize_host, numPivots, 
		    			   sampleSize_host, numBuckets, minimum_host, maximum_host);
		    			   
		    // make any slopes that were infinity due to division by zero (due to no
        	//  difference between the two associated pivots) into zero, so all the
        	//  values which use that slope are projected into a single bucket
		    for (int i = 0; i < numPivots - 1; i++)
		        if (isinf(slopes[i]))
		            slopes[i] = 0;
		    			   
		    // documentation
		    for (int i = 0; i < numPivots - 1; i++) {
		    	pivotsLeft[i] = pivots[i];
		        pivotsRight[i] = pivots[i + 1];
		        kthnumBuckets[i] = numBuckets / (numPivots - 1) * i;
		        
		        // test part
		        // printf("%d, %d\n", pivotsLeft[i], pivotsRight[i]);
		    }
		    numUniqueBuckets = numPivots - 1;
		    
		    /*
		    // test part
		    printf("\n");
		    printf("pivotsLeft:\n");
		    for (int x = 0; x < numUniqueBuckets; x++) {
		    	printf("%u, ", pivotsLeft[x]);
		    }
		    printf("\n");
		    printf("pivotsRight:\n");
		    for (int x = 0; x < numUniqueBuckets; x++) {
		    	printf("%u, ", pivotsRight[x]);
		    }
		    printf("\n");
		    printf("slopes:\n");
		    for (int x = 0; x < numUniqueBuckets; x++) {
		    	printf("%.10lf, ", slopes[x]);
		    }
		    printf("\n");
		    printf("kthnumBuckets:\n");
		    for (int x = 0; x < numUniqueBuckets; x++) {
		    	printf("%u, ", kthnumBuckets[x]);
		    }
		    printf("\n");
		    */
		    
		    
		    
		    // send pivots and kthnumBuckets to all other ranks
		    for (int i = 1; i < RANK_NUM; i++) {
		    	MPI_Send_CALL(&numUniqueBuckets, 1, i, 
		    			 3, MPI_COMM_WORLD);
		    	MPI_Send_CALL(slopes, numUniqueBuckets, i, 
		    			 4, MPI_COMM_WORLD);
		    	MPI_Send_CALL(pivotsLeft, numUniqueBuckets, i, 
		    			 5, MPI_COMM_WORLD);
		    	MPI_Send_CALL(pivotsRight, numUniqueBuckets, i, 
		    			 6, MPI_COMM_WORLD);
		    	MPI_Send_CALL(kthnumBuckets, numUniqueBuckets, i, 
		    			 7, MPI_COMM_WORLD);
		    }
        }
        
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		/// ***********************************************************
        /// **** STEP 2: CreateBuckets
        /// **** STEP 2.3: Declare and Generate Pivots and Slopes
        /// **** All slots receive and save KD intervals
        /// ***********************************************************
		
		if (true) {
			// other ranks receive KD intervals from host
			if (rank != 0) {
				MPI_Recv_CALL(&numUniqueBuckets, 1, 0, 3, MPI_COMM_WORLD,
			     		 MPI_STATUS_IGNORE);
			    MPI_Recv_CALL(slopes, numUniqueBuckets, 0, 4, MPI_COMM_WORLD,
			     		 MPI_STATUS_IGNORE);
			    MPI_Recv_CALL(pivotsLeft, numUniqueBuckets, 0, 5, MPI_COMM_WORLD,
			     		 MPI_STATUS_IGNORE);
			    MPI_Recv_CALL(pivotsRight, numUniqueBuckets, 0, 6, MPI_COMM_WORLD,
			     		 MPI_STATUS_IGNORE);
			    MPI_Recv_CALL(kthnumBuckets, numUniqueBuckets, 0, 7, MPI_COMM_WORLD,
			     		 MPI_STATUS_IGNORE);
			}
			
			// All slots send information to GPU
			cudaMemcpy(d_slopes, slopes, (numPivots - 1) * sizeof(double), 
					   cudaMemcpyHostToDevice);
			cudaMemcpy(d_pivotsLeft, pivotsLeft, numUniqueBuckets * sizeof(T), 
					   cudaMemcpyHostToDevice);
			cudaMemcpy(d_pivotsRight, pivotsRight, numUniqueBuckets * sizeof(T), 
					   cudaMemcpyHostToDevice);
			cudaMemcpy(d_kthnumBuckets, kthnumBuckets, numUniqueBuckets * sizeof(unsigned int), 
					   cudaMemcpyHostToDevice);
		}
		
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		/*
		// test part
		if (rank == 0) {
			printf("rank 0, finishing STEP 2\n");
		}
		if (rank == 2) {
			printf("rank 2, finishing STEP 2\n");
		}
		*/
		
		
/*
		// test part
		if (rank == 2) {
			for (int i = 0; i < numUniqueBuckets; i++) {
				printf("%d\n", pivotsLeft[i]);
			}
		}
*/

		
		
        /// ***********************************************************
        /// **** STEP 3: AssignBuckets
        /// **** Using the function assignSmartBucket
        /// ***********************************************************
        
        if (true) {
		    assignSmartBucket_distributive_CALL
		    		(d_vector_local, length_local, d_elementToBucket, d_slopes, 
					 d_pivotsLeft,  d_pivotsRight, d_kthnumBuckets, d_bucketCount, 
					 numUniqueBuckets, numBuckets, offset, numBlocks, threadsPerBlock);
        }
        
        /*
        // test part
        T* h_vector_test = (T*)malloc(sizeof(T) * length_local);
        unsigned int* h_elementToBucket_test = (unsigned int*)malloc(sizeof(unsigned int) * length_local);
        
        cudaMemcpy(h_vector_test, d_vector_local, sizeof(T) * length_local, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_elementToBucket_test, d_elementToBucket, sizeof(unsigned int) * length_local, cudaMemcpyDeviceToHost);
        
        if (rank == 0) {
        	printf("rank 0\n");
        	for (int x = 0; x < 500; x++) {
        		printf("%d: %f: %u,  ", x, h_vector_test[x], h_elementToBucket_test[x]);
        	}
        	printf("\n");
        }
        else {
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 1) {
        	printf("rank 1\n");
        	for (int x = 0; x < 500; x++) {
        		printf("%d: %f: %u,  ", x, h_vector_test[x], h_elementToBucket_test[x]);
        	}
        	printf("\n");
        }
        else {
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 2) {
        	printf("rank 2\n");
        	for (int x = 0; x < 500; x++) {
        		printf("%d: %f: %u,   ", x, h_vector_test[x], h_elementToBucket_test[x]);
        	}
        	printf("\n");
        }
        else {
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 3) {
        	printf("rank 3\n");
        	for (int x = 0; x < 500; x++) {
        		printf("%d: %f: %u,  ", x, h_vector_test[x], h_elementToBucket_test[x]);
        	}
        	printf("\n");
        }
        else {
        }
        */
        
        
        									
        cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		/*
		// test part
		if (rank == 0) {
			printf("rank 0, finishing STEP 3\n");
		}
		if (rank == 2) {
			printf("rank 2, finishing STEP 3\n");
		}
		*/
		
		
		/*
		// test part
		if (rank == 0) {
			unsigned int * elementToBucket_test = (unsigned int*)malloc(sizeof(unsigned int) * length_local);
			cudaMemcpy(elementToBucket_test, d_elementToBucket, length_local * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			printf("Step 3, elementToBucket:\n");
			for (int i = 0; i < length_local; i++) {
				printf("%d ", elementToBucket_test[i]);
			}
			printf("\n");
		}
		*/
		
		// test part
        // T* h_vector_local_test = (T*)malloc(sizeof(T) * length_local);
        // T* h_elementToBucket_test = (T*)malloc(sizeof(T) * length_local);


		/// ***********************************************************
        /// **** STEP 4: IdentifyActiveBuckets
        /// **** STEP 4.1 Update the bucketCount
        /// **** Each slots calculates bucketCount and
        /// **** send it to the host to cumulate
        /// ***********************************************************
        
        // each slot updates their bucketCount and send information to the host
        if (true) {
        	sumCounts_CALL(d_bucketCount, numBuckets, numBlocks, threadsPerBlock);
        	
        	// potential to simplify using only GPU
        	// consider the last row which holds the total counts
        	int sumRowIndex = numBuckets * (numBlocks - 1);
        	cudaMemcpy(h_bucketCount, d_bucketCount + sumRowIndex, sizeof(unsigned int) * numBuckets,
        			   cudaMemcpyDeviceToHost);
        			   
        	if (rank != 0) {
        		MPI_Send_CALL(h_bucketCount, numBuckets, 0, 
		    			 8, MPI_COMM_WORLD);
        	}
        }
        
        cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
        
        // potential to simplify using GPU
        // host receive information and calculate cumulative sum of bucketCount
        if (rank == 0) {
        	for (int i = 0; i < numBuckets; i++) {
        		h_bucketCount_Host[i] = h_bucketCount[i];
        	}
        	for (int i = 1; i < RANK_NUM; i++) {
        		MPI_Recv_CALL(h_bucketCount_Receive, numBuckets, i, 8, 
        				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        		for (int j = 0; j < numBuckets; j++) {
        			h_bucketCount_Host[j] += h_bucketCount_Receive[j];
        		}
        	}
        }
        
        cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
/*
		// test part
		if (rank == 0) {
			printf("Step 4.1, h_bucketCount:\n");
			for (int i = 0; i < numBuckets; i++) {
				printf("%d ", h_bucketCount[i]);
			}
			printf("\n");
		}
*/

		/// ***********************************************************
        /// **** STEP 4: IdentifyActiveBuckets
        /// **** STEP 4.2 Find and update the kth buckets
        /// **** and their respective indices
        /// ***********************************************************
        
        if (rank == 0) {
        	findKBuckets(h_bucketCount_Host, numBuckets, kVals, numKs, 
        				 kthBucketScanner, kthBuckets, numBlocks);
        				 
        	updatekVals_distributive<T>(kVals, &numKs, output, kIndices, &length_host, 
        					&length_host_Old, h_bucketCount_Host, kthBuckets, kthBucketScanner,
                            reindexCounter, uniqueBuckets, uniqueBucketCounts, 
                            &numUniqueBuckets, &numUniqueBucketsOld, tempKorderBucket, 
                            tempKorderIndeces, &tempKorderLength); 
		  				 
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        /// ***********************************************************
        /// **** STEP 4: IdentifyActiveBuckets
        /// **** STEP 4.3 Find and update output
        /// **** if we find particular order statistics
        /// ***********************************************************
        
        // host send potential k-order statistics information to each slot
        if (rank == 0) {
        	// test part
        	// printf("Rank 0, tempKorderLength: %d\n", tempKorderLength);
        	for (int i = 1; i < RANK_NUM; i++) {
        		MPI_Send_CALL(&tempKorderLength, 1, i, 
							 9, MPI_COMM_WORLD);
				MPI_Send_CALL(&numKs, 1, i, 12, MPI_COMM_WORLD);
        	}
        	
        	if (tempKorderLength > 0) {
        		for (int i = 1; i < RANK_NUM; i++) {
					MPI_Send_CALL(tempKorderBucket, tempKorderLength, i, 
							 10, MPI_COMM_WORLD);
					MPI_Send_CALL(tempKorderIndeces, tempKorderLength, i, 
							 11, MPI_COMM_WORLD);
				}
        	}
        }
        
        
        // each slots receive information and perform updateOutput function
        if (true) {
        	if (rank != 0) {
				MPI_Recv_CALL(&tempKorderLength, 1, 0, 9, MPI_COMM_WORLD,
							 		 MPI_STATUS_IGNORE);
				MPI_Recv_CALL(&numKs, 1, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			
		    if (tempKorderLength > 0) {
		    	if (rank != 0) {
					MPI_Recv_CALL(tempKorderBucket, tempKorderLength, 0, 10, MPI_COMM_WORLD,
					 		 MPI_STATUS_IGNORE);
					MPI_Recv_CALL(tempKorderIndeces, tempKorderLength, 0, 11, MPI_COMM_WORLD,
					 		 MPI_STATUS_IGNORE);
		    	}
		    	
		    	cudaMemcpy(d_tempKorderBucket, tempKorderBucket, 
		    			   tempKorderLength * sizeof(unsigned int), 
		    			   cudaMemcpyHostToDevice);
		    	cudaMemcpy(d_tempKorderIndeces, tempKorderIndeces, 
		    			  tempKorderLength * sizeof(unsigned int), 
		    			   cudaMemcpyHostToDevice);
		    			   
		    	cudaMemset(d_tempOutput, 0.0, tempKorderLength * sizeof(T));
		    			   
		    	updateOutput_distributive_CALL
		    			(d_vector_local, d_elementToBucket, length_local_Old, d_tempOutput, 
		    			 d_tempKorderBucket, tempKorderLength, offset, threadsPerBlock);
		    	
		    	cudaMemcpy(tempOutput, d_tempOutput, tempKorderLength * sizeof(T), 
		    			   cudaMemcpyDeviceToHost);
		    			   
		    	cudaDeviceSynchronize();
		    			   
		    	// each slots send output being already found to the host
		    	if (rank != 0) {
		    		MPI_Send_CALL(tempOutput, tempKorderLength, 0, 13, MPI_COMM_WORLD);
		    	}
		    	
		    	MPI_Barrier(MPI_COMM_WORLD);
				cudaDeviceSynchronize();
				
				/*
				//test part
				if (rank == 0) {
					printf("rank 0, tempOutput\n");
					for (int x = 0; x < tempKorderLength; x++) {
						printf("%u, ", tempOutput[x]);
					}
					printf("\n");
				}
				MPI_Barrier(MPI_COMM_WORLD);
				if (rank == 3) {
					printf("rank 3, tempOutput\n");
					for (int x = 0; x < tempKorderLength; x++) {
						printf("%u, ", tempOutput[x]);
					}
					printf("\n");
				}
				MPI_Barrier(MPI_COMM_WORLD);
				if (rank == 2) {
					printf("rank 2, tempOutput\n");
					for (int x = 0; x < tempKorderLength; x++) {
						printf("%u, ", tempOutput[x]);
					}
					printf("\n");
				}
				MPI_Barrier(MPI_COMM_WORLD);
				if (rank == 1) {
					printf("rank 1, tempOutput\n");
					for (int x = 0; x < tempKorderLength; x++) {
						printf("%u, ", tempOutput[x]);
					}
					printf("\n");
				}
				MPI_Barrier(MPI_COMM_WORLD);	   
		    	cudaDeviceSynchronize();
		    	*/
        	}
        }
        
        
		// the host summarize information and update the output
        if (rank == 0) {
        	if (tempKorderLength > 0) {
        		// receive tempOutput from each slots
        		for (int i = 1; i < RANK_NUM; i++) {
        			MPI_Recv_CALL(tempOutput_Receive, tempKorderLength, i, 13, 
        				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        			for (int j = 0; j < tempKorderLength; j++) {
        				tempOutput[j] += tempOutput_Receive[j];
        			}
        		}
        		
        		// copy it to the output
        		for (int i = 0; i < tempKorderLength; i++) {
        			output[tempKorderIndeces[i]] = tempOutput[i];
        		}
        		
        		
        	}
        }
        
        
        cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
      	
      	
		bool whetherEnterLoop = true;
		if (numKs <= 0)
			whetherEnterLoop = false;
			
		/*
		// test part
		if (rank == 2) {
			printf("numKs: %d\n", numKs);
		}
		*/
		
		/*
		// test part
		if (rank == 0) {
			printf("rank 0, finishing STEP 4\n");
		}
		if (rank == 2) {
			printf("rank 2, finishing STEP 4\n");
		}
		*/
		
		
		int numLengthEqual = 0;
		
		/// ***********************************************************
        /// **** STEP 5: Reduce
        /// **** Iteratively go through the loop to find correct
        /// **** order statistics and reduce the vector size
        /// ***********************************************************
        
        for (int l = 0; l < 40 && whetherEnterLoop; l++) {
        
        	/*
        	// test part
        	if (rank == 0) {
            	printf("This is iteration %d\n", l);
            }
            */
        
        	/// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.1: Copy active elements
            /// **** Copy the elements from the unique active buckets
            /// ***********************************************************
            
            
            /// ***********************************************************
            /// **** Step 5.1.1: update reindexCounter
            /// ***********************************************************
        	
        	// host send uniqueBuckets to each slots
        	if (rank == 0) {
        		for (int i = 1; i < RANK_NUM; i++) {
        			MPI_Send_CALL(&numUniqueBuckets, 1, i, 
								  50, MPI_COMM_WORLD);
		    		MPI_Send_CALL(uniqueBuckets, numUniqueBuckets, i, 
								  51, MPI_COMM_WORLD);
				}
        	}
        	
        	if (true) {
        		// update reindexCounts on each slots firstly
        		if (rank != 0) {
		    		MPI_Recv_CALL(&numUniqueBuckets, 1, 0, 50, MPI_COMM_WORLD,
						 		 MPI_STATUS_IGNORE);
        			MPI_Recv_CALL(uniqueBuckets, numUniqueBuckets, 0, 51, MPI_COMM_WORLD,
					 		 MPI_STATUS_IGNORE);
        		}
        		
        		updateReindexCounter_distributive
        					(reindexCounter, h_bucketCount, uniqueBuckets, &length_local,
        					 &length_local_Old, numUniqueBuckets);
        		
        		cudaMemcpy(d_reindexCounter, reindexCounter, numUniqueBuckets * sizeof(unsigned int), 
        				   cudaMemcpyHostToDevice);
        		cudaMemcpy(d_uniqueBuckets, uniqueBuckets, numUniqueBuckets * sizeof(unsigned int), 
        				   cudaMemcpyHostToDevice);
        				   
        		reindexCounts_CALL(d_bucketCount, numBuckets, numBlocks, d_reindexCounter, 
        						   d_uniqueBuckets, numUniqueBuckets, threadsPerBlock);
        	}
        	
        	cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			
			
			/// ***********************************************************
            /// **** Step 5.1.2: copy active elements
            /// ***********************************************************
        
        	// each slots copy the active elements
        	if (true) {
        		copyElements_distributive_CALL
        			(d_vector_local, d_newvector, length_local_Old, d_elementToBucket, 
        			 d_uniqueBuckets, numUniqueBuckets, d_bucketCount, 
        			 numBuckets, offset, threadsPerBlock, numBlocks);
        			 
        		swapPointers(&d_vector_local, &d_newvector);
        	}
        
        	cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			
			/*
			// test part
			for (int j = 0; j < RANK_NUM; j++) {
				if (rank == j) {
					T* vector_local = (T*)malloc(length_local * sizeof(T));
					cudaMemcpy(vector_local, d_vector_local, length_local * sizeof(T), 
							   cudaMemcpyDeviceToHost);
					
					printf("iteration %d, rank %d d_vector_local:\n", l, j);
					for (int i = 0; i < length_local; i++) {
						printf("%d ", vector_local[i]);
					}
					printf("\n");
					
				}
			}
			*/
			
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			
			/*
			// test part
			if (rank == 0) {
		        printf("numKs: %d, length: %d, numUniqueBuckets: %d, tempKorderLength: %d\n", numKs, length_host, numUniqueBuckets, tempKorderLength);
		        printf("lengthOld: %d, numUniqueBucketsOld: %d\n", length_host_Old, numUniqueBucketsOld);
            }
            */
			
			
        
        
        

            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.2: Update the pivots
            /// **** Update pivots to generate Pivots and Slopes in Step 5.3
            /// ***********************************************************
            
        
        	if (rank == 0) {
        		updatePivots_distributive_CALL
        			(d_pivotsLeft, d_newPivotsLeft, d_newPivotsRight,
                     d_slopes, d_kthnumBuckets, d_uniqueBuckets,
                     numUniqueBuckets, numUniqueBucketsOld, offset,
                     threadsPerBlock);
                
                swapPointers(&d_pivotsLeft, &d_newPivotsLeft);
            	swapPointers(&d_pivotsRight, &d_newPivotsRight);
        	}
        	
        	
        	cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			
			
        	
        	/// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.3: create slopes and buckets offset
            /// **** create slopes and buckets offset for next iteration
            /// ***********************************************************
            
            /// ***********************************************************
            /// **** Step 5.3.1: create slopes and bucket offset
            /// ***********************************************************
        	
        	if (rank == 0) {
        		cudaMemcpy(d_uniqueBucketCounts, uniqueBucketCounts, 
        				   numUniqueBuckets * sizeof(unsigned int), cudaMemcpyHostToDevice);
        				   
        		generateBucketsandSlopes_distributive_CALL
        			(d_pivotsLeft, d_pivotsRight, d_slopes, d_uniqueBucketCounts,
                     numUniqueBuckets, d_kthnumBuckets, length_host, offset, 
                     numBuckets, threadsPerBlock);
        			 
        	}
        	
        	cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
        	
        	
        	/// ***********************************************************
            /// **** Step 5.3.2: send all info to each slots
            /// ***********************************************************
            
        	if (rank == 0) {
        		// copy back information from kernel
        		cudaMemcpy(slopes, d_slopes, numUniqueBuckets * sizeof(double), 
        				   cudaMemcpyDeviceToHost);
        		cudaMemcpy(pivotsLeft, d_pivotsLeft, numUniqueBuckets * sizeof(T), 
        				   cudaMemcpyDeviceToHost);
        		cudaMemcpy(pivotsRight, d_pivotsRight, numUniqueBuckets * sizeof(T), 
        				   cudaMemcpyDeviceToHost);
        		cudaMemcpy(kthnumBuckets, d_kthnumBuckets, numUniqueBuckets * sizeof(unsigned int),
        				   cudaMemcpyDeviceToHost);
        		
        		// send it to each slots
        		for (int i = 1; i < RANK_NUM; i++) {
					MPI_Send_CALL(slopes, numUniqueBuckets, i, 
							 52, MPI_COMM_WORLD);
					MPI_Send_CALL(pivotsLeft, numUniqueBuckets, i, 
							 53, MPI_COMM_WORLD);
					MPI_Send_CALL(pivotsRight, numUniqueBuckets, i, 
							 54, MPI_COMM_WORLD);
					MPI_Send_CALL(kthnumBuckets, numUniqueBuckets, i, 
							 55, MPI_COMM_WORLD);
		    	}
        	}
        	
        	
        	// each slot receive information and send it to kernel
        	if (true) {
    			if (rank != 0) {
					MPI_Recv_CALL(slopes, numUniqueBuckets, 0, 52, MPI_COMM_WORLD,
					 		 MPI_STATUS_IGNORE);
					MPI_Recv_CALL(pivotsLeft, numUniqueBuckets, 0, 53, MPI_COMM_WORLD,
					 		 MPI_STATUS_IGNORE);
					MPI_Recv_CALL(pivotsRight, numUniqueBuckets, 0, 54, MPI_COMM_WORLD,
					 		 MPI_STATUS_IGNORE);
					MPI_Recv_CALL(kthnumBuckets, numUniqueBuckets, 0, 55, MPI_COMM_WORLD,
					 		 MPI_STATUS_IGNORE);
				}
				
			
				// All slots send information to GPU
				cudaMemcpy(d_slopes, slopes, numUniqueBuckets * sizeof(double), 
						   cudaMemcpyHostToDevice);
				cudaMemcpy(d_pivotsLeft, pivotsLeft, numUniqueBuckets * sizeof(T), 
						   cudaMemcpyHostToDevice);
				cudaMemcpy(d_pivotsRight, pivotsRight, numUniqueBuckets * sizeof(T), 
						   cudaMemcpyHostToDevice);
				cudaMemcpy(d_kthnumBuckets, kthnumBuckets, 
						   numUniqueBuckets * sizeof(unsigned int), 
						   cudaMemcpyHostToDevice);
		    	
		    }
        
        	cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			
			
			
            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.4: assign buckets
            /// **** assign elements to correct buckets in iteration
            /// ***********************************************************		
			
			if (true) {
				assignSmartBucket_distributive_CALL
						(d_vector_local, length_local, d_elementToBucket, d_slopes, 
						 d_pivotsLeft,  d_pivotsRight, d_kthnumBuckets, d_bucketCount, 
						 numUniqueBuckets, numBuckets, offset, numBlocks, threadsPerBlock);
        	}
        	
        	/*
        	// test part
			T* h_vector_test = (T*)malloc(sizeof(T) * length_local);
			unsigned int* h_elementToBucket_test = (unsigned int*)malloc(sizeof(unsigned int) * length_local);
			
			cudaMemcpy(h_vector_test, d_vector_local, sizeof(T) * length_local, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_elementToBucket_test, d_elementToBucket, sizeof(unsigned int) * length_local, cudaMemcpyDeviceToHost);
			
			if (rank == 0 && l == 0) {
				printf("rank 0\n");
				for (int x = 0; x < 500; x++) {
					printf("%d: %f: %u,  ", x, h_vector_test[x], h_elementToBucket_test[x]);
				}
				printf("\n");
			}
			else {
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 1 && l == 0) {
				printf("rank 1\n");
				for (int x = 0; x < 500; x++) {
					printf("%d: %f: %u,  ", x, h_vector_test[x], h_elementToBucket_test[x]);
				}
				printf("\n");
			}
			else {
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 2 && l == 0) {
				printf("rank 2\n");
				for (int x = 0; x < 500; x++) {
					printf("%d: %f: %u,   ", x, h_vector_test[x], h_elementToBucket_test[x]);
				}
				printf("\n");
			}
			else {
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 3 && l == 0) {
				printf("rank 3\n");
				for (int x = 0; x < 500; x++) {
					printf("%d: %f: %u,  ", x, h_vector_test[x], h_elementToBucket_test[x]);
				}
				printf("\n");
			}
			else {
			}
			*/
		
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			
			
			
			
			/// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.5: IdentifyActiveBuckets
            /// **** Find kth buckets and update their respective indices
            /// ***********************************************************
            
            
            /// ***********************************************************
            /// **** Step 5.5.1: Each slots update and send bucketCount
            /// ***********************************************************
			
			// each slot updates their bucketCount and send information to the host
		    if (true) {
		    	sumCounts_CALL(d_bucketCount, numBuckets, numBlocks, threadsPerBlock);
		    	
		    	// potential to simplify using only GPU
		    	// consider the last row which holds the total counts
		    	int sumRowIndex = numBuckets * (numBlocks - 1);
		    	cudaMemcpy(h_bucketCount, d_bucketCount + sumRowIndex, 
		    			   sizeof(unsigned int) * numBuckets,
		    			   cudaMemcpyDeviceToHost);
		    			   
		    	if (rank != 0) {
		    		MPI_Send_CALL(h_bucketCount, numBuckets, 0, 
							 56, MPI_COMM_WORLD);
		    	}
		    }
		    
		    cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
		    
		    // potential to simplify using GPU
		    // host receive information and calculate cumulative sum of bucketCount
		    if (rank == 0) {
		    	for (int i = 0; i < numBuckets; i++) {
		    		h_bucketCount_Host[i] = h_bucketCount[i];
		    	}
		    	for (int i = 1; i < RANK_NUM; i++) {
		    		MPI_Recv_CALL(h_bucketCount_Receive, numBuckets, i, 56, 
		    				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    		for (int j = 0; j < numBuckets; j++) {
		    			h_bucketCount_Host[j] += h_bucketCount_Receive[j];
		    		}
		    	}
		    }
		    
		    cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
		
            /// ***********************************************************
            /// **** Step 5.5.2: Find and update the kth buckets
            /// ***********************************************************		
		
		    if (rank == 0) {
		    	findKBuckets(h_bucketCount_Host, numBuckets, kVals, numKs, 
		    				 kthBucketScanner, kthBuckets, numBlocks);
		    				 
		    	updatekVals_distributive<T>(kVals, &numKs, output, kIndices, &length_host, 
		    					&length_host_Old, h_bucketCount_Host, kthBuckets, kthBucketScanner,
		                        reindexCounter, uniqueBuckets, uniqueBucketCounts, 
		                        &numUniqueBuckets, &numUniqueBucketsOld, tempKorderBucket, 
		                        tempKorderIndeces, &tempKorderLength); 
			  				 
		    }
		    
		    MPI_Barrier(MPI_COMM_WORLD);
		    
		    /// ***********************************************************
            /// **** Step 5.5.3: Find and update output
            /// ***********************************************************	
		
			// host send potential k-order statistics information to each slot
		    if (rank == 0) {
		    	// test part
		    	// printf("Rank 0, tempKorderLength: %d\n", tempKorderLength);
		    	for (int i = 1; i < RANK_NUM; i++) {
		    		MPI_Send_CALL(&tempKorderLength, 1, i, 
								 57, MPI_COMM_WORLD);
					MPI_Send_CALL(&numKs, 1, i, 60, MPI_COMM_WORLD);
					MPI_Send_CALL(&length_host, 1, i, 61, MPI_COMM_WORLD);
					MPI_Send_CALL(&length_host_Old, 1, i, 62, MPI_COMM_WORLD);
		    	}
		    	
		    	if (tempKorderLength > 0) {
		    		for (int i = 1; i < RANK_NUM; i++) {
						MPI_Send_CALL(tempKorderBucket, tempKorderLength, i, 
								 58, MPI_COMM_WORLD);
						MPI_Send_CALL(tempKorderIndeces, tempKorderLength, i, 
								 59, MPI_COMM_WORLD);
						
					}
		    	}
		    }
		    
		    cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
		    
		    
		    // each slots receive information and perform updateOutput function
		    if (true) {
		    	if (rank != 0) {
					MPI_Recv_CALL(&tempKorderLength, 1, 0, 57, MPI_COMM_WORLD,
								 		 MPI_STATUS_IGNORE);
					MPI_Recv_CALL(&numKs, 1, 0, 60, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv_CALL(&length_host, 1, 0, 61, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv_CALL(&length_host_Old, 1, 0, 62, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				
				if (tempKorderLength > 0) {
					if (rank != 0) {
						MPI_Recv_CALL(tempKorderBucket, tempKorderLength, 0, 58, 
								      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						MPI_Recv_CALL(tempKorderIndeces, tempKorderLength, 0, 59, 
									  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					}
					
					cudaMemcpy(d_tempKorderBucket, tempKorderBucket, 
							   tempKorderLength * sizeof(unsigned int), 
							   cudaMemcpyHostToDevice);
					cudaMemcpy(d_tempKorderIndeces, tempKorderIndeces, 
							   tempKorderLength * sizeof(unsigned int), 
							   cudaMemcpyHostToDevice);
							   
					cudaMemset(d_tempOutput, 0.0, tempKorderLength * sizeof(T));
							   
					updateOutput_distributive_CALL
							(d_vector_local, d_elementToBucket, length_local, d_tempOutput, 
							 d_tempKorderBucket, tempKorderLength, offset, threadsPerBlock);
					
					cudaMemcpy(tempOutput, d_tempOutput, tempKorderLength * sizeof(T), 
							   cudaMemcpyDeviceToHost);
					
					/*
					//test part
					if (rank == 0) {
						printf("This is iteration %d\n", l);
						printf("rank 0, tempOutput\n");
						for (int x = 0; x < tempKorderLength; x++) {
							printf("%u, ", tempOutput[x]);
						}
						printf("\n");
					}
					MPI_Barrier(MPI_COMM_WORLD);
					if (rank == 3) {
						printf("rank 3, tempOutput\n");
						for (int x = 0; x < tempKorderLength; x++) {
							printf("%u, ", tempOutput[x]);
						}
						printf("\n");
					}
					MPI_Barrier(MPI_COMM_WORLD);
					if (rank == 2) {
						printf("rank 2, tempOutput\n");
						for (int x = 0; x < tempKorderLength; x++) {
							printf("%u, ", tempOutput[x]);
						}
						printf("\n");
					}
					MPI_Barrier(MPI_COMM_WORLD);
					if (rank == 1) {
						printf("rank 1, tempOutput\n");
						for (int x = 0; x < tempKorderLength; x++) {
							printf("%u, ", tempOutput[x]);
						}
						printf("\n");
					}
					MPI_Barrier(MPI_COMM_WORLD);
					*/
							   
					cudaDeviceSynchronize();
							   
					// each slots send output being already found to the host
					if (rank != 0) {
						MPI_Send_CALL(tempOutput, tempKorderLength, 0, 63, MPI_COMM_WORLD);
					}
		    	}
		    }
		    
		    cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
		    
			// the host summarize information and update the output
		    if (rank == 0) {
		    	if (tempKorderLength > 0) {
		    		// receive tempOutput from each slots
		    		for (int i = 1; i < RANK_NUM; i++) {
		    			MPI_Recv_CALL(tempOutput_Receive, tempKorderLength, i, 63, 
		    				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    			for (int j = 0; j < tempKorderLength; j++) {
		    				tempOutput[j] += tempOutput_Receive[j];
		    			}
		    		}
		    		
		    		// copy it to the output
		    		for (int i = 0; i < tempKorderLength; i++) {
		    			output[tempKorderIndeces[i]] = tempOutput[i];
		    		}
		    	}
		    }
		    
		    cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			
			/*
			// test part
			if (rank == 0) {
				printf("rank 0, length_host: %d, length_host_Old: %d\n", length_host, length_host_Old);
			}
			if (rank == 2) {
				printf("rank 2, length_host: %d, length_host_Old: %d\n", length_host, length_host_Old);
			}
			*/
			
			
			
			if (length_host == length_host_Old) {
				numLengthEqual++;
				if (numLengthEqual > 2 || length_host == 0 || numKs == 0) {
					break;
				}
			}
			else {
            	numLengthEqual = 0;
            }
			
			
		}
		
		/*
		// test part
		if (rank == 0) {
			printf("rank 0, finishing STEP 5\n");
		}
		if (rank == 2) {
			printf("rank 2, finishing STEP 5\n");
		}
		*/
		
		
		
		/// ***********************************************************
        /// **** STEP 6: Finalize
        /// **** Update repeated k-order statistics to output and 
        /// **** free all the memory
        /// ***********************************************************
        
        /*
        // test part
        if (rank == 0) {
        	printf("from rank 0, tempKorderIndices:\n");
        	for (int i = 0; i < numKs; i++) {
        		printf("%d ", tempKorderIndeces[i]);
        	}
        	printf("\n");
        }
        */
        
        /*
        // test part
        // printf("Done iteration, numKs: %d from rank %d\n", numKs, rank);
        
        cudaMemcpy(h_vector_local_test, d_vector_local, sizeof(T) * length_local, cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_elementToBucket_test, d_elementToBucket, sizeof(T) * length_local, cudaMemcpyDeviceToHost);
	    
	    cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		// test part
		if (rank == 0) {
			printf("rank 0, d_vector_local and d_elementToBucket:\n");
			for (int x = 0; x < length_local; x++) {
				printf("%u: %u,  ", h_vector_local_test[x], h_elementToBucket_test[x]);
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 1) {
			printf("rank 1, d_vector_local and d_elementToBucket:\n");
			for (int x = 0; x < length_local; x++) {
				printf("%u: %u,  ", h_vector_local_test[x], h_elementToBucket_test[x]);
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 2) {
			printf("rank 2, d_vector_local and d_elementToBucket:\n");
			for (int x = 0; x < length_local; x++) {
				printf("%u: %u,  ", h_vector_local_test[x], h_elementToBucket_test[x]);
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 3) {
			printf("rank 3, d_vector_local and d_elementToBucket:\n");
			for (int x = 0; x < length_local; x++) {
				printf("%u: %u,  ", h_vector_local_test[x], h_elementToBucket_test[x]);
			}
			printf("\n");
		}
		*/
		
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		// deal with the repeated cases
		if (numKs > 0) {
			if (rank == 0) {
				for (int i = 1; i < RANK_NUM; i++) {
					MPI_Send_CALL(kthBuckets, numKs, i, 64, MPI_COMM_WORLD);
				}
			}
			
			MPI_Barrier(MPI_COMM_WORLD);
			
			if (rank != 0) {
				MPI_Recv_CALL(kthBuckets, numKs, 0, 64, MPI_COMM_WORLD,
							  MPI_STATUS_IGNORE);
			}
			
			cudaMemcpy(d_kthBuckets, kthBuckets, numKs * sizeof(unsigned int),
					   cudaMemcpyHostToDevice);
					   
			cudaMemset(d_tempOutput, 0.0, numKs * sizeof(T));
			
			updateOutput_distributive_CALL
					(d_vector_local, d_elementToBucket, length_local, d_tempOutput, 
					 d_kthBuckets, numKs, offset, threadsPerBlock);
					 
			cudaMemcpy(tempOutput, d_tempOutput, numKs * sizeof(T), 
					   cudaMemcpyDeviceToHost);
					   
			MPI_Barrier(MPI_COMM_WORLD);
			cudaDeviceSynchronize();
			
			/*  
			//test part
			if (rank == 0) {
				printf("Done iteration\n");
				printf("rank 0, tempOutput\n");
				for (int x = 0; x < numKs; x++) {
					printf("%u, ", tempOutput[x]);
				}
				printf("\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 3) {
				printf("rank 3, tempOutput\n");
				for (int x = 0; x < numKs; x++) {
					printf("%u, ", tempOutput[x]);
				}
				printf("\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 2) {
				printf("rank 2, tempOutput\n");
				for (int x = 0; x < numKs; x++) {
					printf("%u, ", tempOutput[x]);
				}
				printf("\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 1) {
				printf("rank 1, tempOutput\n");
				for (int x = 0; x < numKs; x++) {
					printf("%u, ", tempOutput[x]);
				}
				printf("\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
			*/
							  
					   
			
			// each slots send output being already found to the host
			if (rank != 0) {
				MPI_Send_CALL(tempOutput, numKs, 0, 65, MPI_COMM_WORLD);
			}
			
			MPI_Barrier(MPI_COMM_WORLD);
			
			if (rank == 0) {
	    		// receive tempOutput from each slots
	    		for (int i = 1; i < RANK_NUM; i++) {
	    			MPI_Recv_CALL(tempOutput_Receive, numKs, i, 65, 
	    				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    			for (int j = 0; j < numKs; j++) {
	    				if (absolute(tempOutput_Receive[j]) > absolute(tempOutput[j]))
	    					tempOutput[j] = tempOutput_Receive[j];
	    			}
	    		}
	    		
	    		// copy it to the output
	    		for (int i = 0; i < numKs; i++) {
	    			output[kIndices[i]] = tempOutput[i];
	    		}	
		    }
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		cudaDeviceSynchronize();
		
		
		
		
		// free all the memory
		free(slopes);
        free(pivots);
        free(pivotsLeft);
        free(pivotsRight);
        free(h_bucketCount);
        free(kthBuckets);
        free(kthBucketScanner);
        free(kIndices);
        free(uniqueBuckets);
        free(uniqueBucketCounts);
        free(reindexCounter);
        free(kthnumBuckets);
        free(tempOutput);
        free(tempKorderBucket);
        free(tempKorderIndeces);
        free(sampleVector);


        cudaFree(d_slopes);
        cudaFree(d_pivots);
        cudaFree(d_pivotsLeft);
        cudaFree(d_pivotsRight);
        cudaFree(d_newPivotsLeft);
        cudaFree(d_newPivotsRight);
        cudaFree(d_elementToBucket);
        cudaFree(d_bucketCount);
        cudaFree(addressOfd_newvector);
        cudaFree(d_kVals);
        cudaFree(d_kthBuckets);
        cudaFree(d_kIndices);
        cudaFree(d_uniqueBuckets);
        cudaFree(d_uniqueBucketCounts);
        cudaFree(d_reindexCounter);
        cudaFree(d_kthnumBuckets);
        cudaFree(d_tempOutput);
        cudaFree(d_tempKorderBucket);
        cudaFree(d_tempKorderIndeces);
        cudaFree(d_sampleVector);
        
        if (rank == 0) {
        	free(h_bucketCount_Host);
        	free(h_bucketCount_Receive);
        	free(tempOutput_Receive);
        	free(sampleVector_Host);
        	cudaFree(d_sampleVector_Host);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
		cudaDeviceSynchronize();
	
		return 0;
	}
	
	template int distributedSMOS 
			(int* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, int* output, int blocks, int threads, int numBuckets, int numPivots, 
		     int rank);
	template unsigned int distributedSMOS 
			(unsigned int* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, unsigned int* output, int blocks, int threads, int numBuckets, int numPivots, 
		     int rank);
	template float distributedSMOS 
			(float* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, float* output, int blocks, int threads, int numBuckets, int numPivots, 
		     int rank);
	template double distributedSMOS 
			(double* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, double* output, int blocks, int threads, int numBuckets, int numPivots, 
		     int rank);
		     
		     
		     
	template <typename T>
	T distributedSMOSWrapper (T* d_vector_local, int length_local, unsigned int * kVals_ori, int numKs,
                          T* output, int blocks, int threads, int rank) {

		int numBuckets = 8192;
		unsigned int * kVals = (unsigned int *)malloc(numKs * sizeof(unsigned int));
		
		// get the total number of vector
		int length_total = length_local;
		int length_each_receive;
		
		if (rank != 0) {
		    	MPI_Send_CALL(&length_local, 1, 0, 1001, MPI_COMM_WORLD);
		}
		
		if (rank == 0) {
			for (int i = 1; i < RANK_NUM; i++) {
				MPI_Recv_CALL(&length_each_receive, 1, i, 1001, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
	     		length_total += length_each_receive;
			}
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		if (rank == 0) {
			for (int i = 1; i < RANK_NUM; i++) {
				MPI_Send_CALL(&length_total, 1, i, 1002, MPI_COMM_WORLD);
			}
		}
		
		
		if (rank != 0) {
		    	MPI_Recv_CALL(&length_total, 1, 0, 1002, MPI_COMM_WORLD, 
					     MPI_STATUS_IGNORE);
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		// turn it into kth smallest
		for (register int i = 0; i < numKs; i++) {
		    kVals[i] = length_total - kVals_ori[i] + 1;
		    output[i] = 0;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		cudaDeviceSynchronize();

		distributedSMOS(d_vector_local, length_local, length_total, kVals, numKs, output, blocks, threads, numBuckets, 17, rank);

		MPI_Barrier(MPI_COMM_WORLD);
		cudaDeviceSynchronize();
		
		/*
		// test part
		if (rank == 0) {
        	printf("k[414]: %.10f\n", output[414]);
		}
		*/

		free(kVals);
		
		MPI_Barrier(MPI_COMM_WORLD);
		cudaDeviceSynchronize();

		return 1;
	}
	
	template int distributedSMOSWrapper 
			(int* d_vector_local, int length_local, unsigned int * kVals_ori, int numKs,
             int* output, int blocks, int threads, int rank);
    template unsigned int distributedSMOSWrapper 
			(unsigned int* d_vector_local, int length_local, unsigned int * kVals_ori, int numKs,
             unsigned int* output, int blocks, int threads, int rank);
    template float distributedSMOSWrapper 
			(float* d_vector_local, int length_local, unsigned int * kVals_ori, int numKs,
             float* output, int blocks, int threads, int rank);
    template double distributedSMOSWrapper 
			(double* d_vector_local, int length_local, unsigned int * kVals_ori, int numKs,
             double* output, int blocks, int threads, int rank);
	
}



/*


/// ***********************************************************
/// ***********************************************************
/// **** distributedSMOS: the main function
/// ***********************************************************
/// ***********************************************************

int main(int argc, char *argv[]) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	/// ***************************************************************
	/// **** Create Problems
	/// **** Create the vector and distribute them to different slots
	/// ***************************************************************

	int vector_size_per_slot = PROBLEM_SIZE / RANK_NUM;
	int* h_vector_total_host;
	int* h_vector_each_send;

	int length_local = PROBLEM_SIZE / RANK_NUM;
	int* h_vector_local = (int*)malloc(sizeof(int) * vector_size_per_slot);
	int* d_vector_local = NULL;


	int numKs = NUM_K_SIZE;
	unsigned int* kVals = (unsigned int*)malloc(numKs * sizeof(unsigned int));

	int* output = (int*)malloc(numKs * sizeof(int));

	int threadsPerBlock = 1024;
	int numBlocks = 12;
	int numBuckets = 8192;

	

	if (rank == 0) {
		h_vector_total_host = (int*)malloc(sizeof(int) * PROBLEM_SIZE);
		h_vector_each_send = (int*)malloc(sizeof(int) * vector_size_per_slot);
		
		time_t t;
		srand((unsigned) time(&t));

		// assign and send the subarray to each slot
		for (int i = 0; i < PROBLEM_SIZE; i++) {
			h_vector_total_host[i] = 1 + i;
		}

		for (int i = 1; i < RANK_NUM; i++) {
			for (int j = 0; j < vector_size_per_slot; j++) {
				h_vector_each_send[j] = h_vector_total_host[j + i * vector_size_per_slot];
			}
			MPI_Send_CALL(h_vector_each_send, vector_size_per_slot,
				i, 0, MPI_COMM_WORLD);
		}

			// assign vector to itself
			for (int i = 0; i < vector_size_per_slot; i++) {
				h_vector_local[i] = h_vector_total_host[i];
			}
	}



    	MPI_Barrier(MPI_COMM_WORLD);
    
	if (true) {
		cudaMalloc((void**)&d_vector_local, sizeof(int) * length_local);

		cudaDeviceSynchronize();

		if (rank != 0) {
		    MPI_Recv(h_vector_local, length_local, MPI_INT, 0, 0, MPI_COMM_WORLD,
			     MPI_STATUS_IGNORE);
		}

		cudaMemcpy(d_vector_local, h_vector_local, sizeof(int) * length_local,
			   cudaMemcpyHostToDevice);
			   
		cudaDeviceSynchronize();


		for (int i = 0; i < numKs; i++) {
			kVals[i] = i * 10 + 1;
			output[i] = 0;
		}

    }

    MPI_Barrier(MPI_COMM_WORLD);

    
    
	DistributedSMOS::distributedSMOS
		(d_vector_local, length_local, kVals, numKs, output, 
		 numBlocks, threadsPerBlock, numBuckets, 17, rank);


	MPI_Barrier(MPI_COMM_WORLD);
	
	
	// test part
	if (rank == 0) {
		for (int i = 0; i < numKs; i++) {
			printf("%d: %d  ", kVals[i], output[i]);
		}
		printf("\n");
	}

	free(h_vector_local);
	free(kVals);
	free(output);
	
	cudaFree(d_vector_local);
	
	if (rank == 0) {
		free(h_vector_total_host);
		free(h_vector_each_send);
	}
	
	
	MPI_Finalize();

	return 0;
}
*/

