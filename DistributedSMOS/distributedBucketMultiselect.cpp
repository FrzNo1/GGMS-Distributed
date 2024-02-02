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
#include "distributedBucketMultiselect_Kernel.cuh"

// inclusde h files
#include "distributedBucketMultiselect.hpp"

#define RANK_NUM 4
#define PROBLEM_SIZE 10000000
#define NUM_K_SIZE 200

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
	
	21:
	22:
	23:
	
	
	
	
	30: each slot send length_local to host in distributedSMOS, STEP 1.1
	31: host send maximum_host to each slot in distributedSMOS, STEP 1.1
	32: host send minimum_host to each slot in distributedSMOS, STEP 1.1


*/

/// ***********************************************************
/// ***********************************************************
/// **** MPI Function Libraries
/// ***********************************************************
/// ***********************************************************

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
		
		
namespace DistributedBucketMultiselect {

	/// ***********************************************************
	/// ***********************************************************
	/// **** distributedBucketMultiselect: the main algorithm
	/// ***********************************************************
	/// ***********************************************************


	/**
	 * This function is the main process of the algorithm. 
	 * It reduces the given multi-selection problem to a smaller 
	 * problem by using bucketing ideas with multiple computers.
	*/
	template <typename T>
	T distributedBucketMultiselect (T* d_vector_local, int length_local,
		int length_total, unsigned int * kVals,  int numKs, T* output, 
		int blocks, int threads, int numBuckets, int numPivots, int rank) {
		
		
		/// ***********************************************************
		/// **** STEP 1: Initialization
		/// **** STEP 1.1: Find Min and Max of the whole vector
		/// **** Find Min and Max in each slot and send it to parent
		/// **** We don't need to go through the rest of the algorithm if it's flat
		/// ***********************************************************
		
		T maximum_local, minimum_local;
		T maximum_host, minimum_host;
		T maximum_each_receive, minimum_each_receive;
		
		if (true) {
			minmax_element_CALL_B(d_vector_local, length_local, &maximum_local, &minimum_local);

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
		
		/*
		// test part
		if (rank == 0) {
			T* vector_local = (T*)malloc(length_local * sizeof(T));
			cudaMemcpy(vector_local, d_vector_local, length_local * sizeof(T),
					   cudaMemcpyDeviceToHost);
			for (int i = 0; i < 200; i++) {
				printf("%f  ", vector_local[i]);
			
			}
			printf("\n");
		
		}
		*/
		
		
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
		int sampleSize_host = 8192;
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
		
		T * pivotsLeft = (T*)malloc(numSpaceAllocate * sizeof(T));
        T * pivotsRight = (T*)malloc(numSpaceAllocate * sizeof(T));
        T * d_pivotsLeft;
        T * d_pivotsRight;
        cudaMalloc(&d_pivotsLeft, numSpaceAllocate * sizeof(T));
        cudaMalloc(&d_pivotsRight, numSpaceAllocate * sizeof(T));
        
		
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
        unsigned int* d_kVals;
        unsigned int* kthBuckets = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int));       	
		unsigned int* d_kthBuckets;   					// may not be used
		unsigned int* kthBucketScanner = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int));
        unsigned int* kIndices = (unsigned int*)malloc(numKs * sizeof(unsigned int));
		unsigned int* d_kIndices;
		unsigned int* uniqueBuckets = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int));    	
		unsigned int* d_uniqueBuckets;  
        unsigned int* uniqueBucketCounts = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int)); // may not be used
		unsigned int* d_uniqueBucketCounts;    		// may not be used
		unsigned int* reindexCounter = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int));   	
		unsigned int* d_reindexCounter;
		unsigned int* kthnumBuckets = (unsigned int*)malloc(numSpaceAllocate * sizeof(unsigned int));
        unsigned int* d_kthnumBuckets;
        cudaMalloc(&d_kVals, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_kIndices, numKs * sizeof(unsigned int));
		cudaMalloc(&d_kthBuckets, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_uniqueBuckets, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_uniqueBucketCounts, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_reindexCounter, numSpaceAllocate * sizeof(unsigned int));
		cudaMalloc(&d_kthnumBuckets, numSpaceAllocate * sizeof(unsigned int));
		
		for (int i = 0; i < numKs; i++) {
			kIndices[i] = i;
		}
		
		// arrays for host to receive information
		unsigned int * h_bucketCount_Host = NULL;
		unsigned int * h_bucketCount_Receive = NULL;
        
        
        
        // variable to store the end result
    	int newInputLength;
    	T* newInput;
    	T * sampleVector_Host = NULL;
		T * d_sampleVector_Host = NULL;
		
		if (rank == 0) {
			h_bucketCount_Host = (unsigned int *)malloc(numBuckets * sizeof(unsigned int));
			h_bucketCount_Receive = (unsigned int *)malloc(numBuckets * sizeof(unsigned int));
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

			sort_by_key_CALL_B(d_kVals, d_kIndices, numKs);
			
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);

		    cudaMemcpy(kIndices, d_kIndices, numKs * sizeof (unsigned int), cudaMemcpyDeviceToHost);
		    cudaMemcpy(kVals, d_kVals, numKs * sizeof (unsigned int), cudaMemcpyDeviceToHost);
		}
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		/*
		int kMaxIndex = numKs - 1;
		int kOffsetMax = 0;
		int kOffsetMin = 0;
		if (rank == 0) {
			printf("%d, %d, %d\n", kVals[0], kVals[1], kVals[2]);
			printf("%d, %d, %d\n", kVals[numKs - 3], kVals[numKs - 2], kVals[numKs - 1]);
			int kMaxIndex = numKs - 1;
			int kOffsetMax = 0;
			while (kVals[kMaxIndex] == length_host) {
			  output[kIndices[numKs-1]] = maximum_host;
			  numKs--;
			  kMaxIndex--;
			  kOffsetMax++;
			}

			while (kVals[0] == 1) {
			  output[kIndices[0]] = minimum_host;
			  kIndices++;
			  kVals++;
			  numKs--;
			  kOffsetMin++;
			}
		}
		
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		*/
		
		
		
		/// ***********************************************************
        /// **** STEP 2: CreateBuckets
        /// **** STEP 2.1: Collect sample
        /// **** Collect samples from each rank
        /// ***********************************************************
        
        
        // randomly select numbers from the all ranks
        if (true) {
		    generateSamples_distributive_CALL_B
				(d_vector_local, d_sampleVector, length_local, sampleSize_local);
				
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
		
		/*
		// test part
		if (rank == 0) {
			for (int i = 0; i < 100; i++) {
				printf("%u  ", sampleVector_Host[i]);
			}
			printf("\n");
			for (int i = 4000; i < 4100; i++) {
				printf("%u  ", sampleVector_Host[i]);
			}
			printf("\n");
			for (int i = sampleSize_host - 100; i < sampleSize_host; i++) {
				printf("%u  ", sampleVector_Host[i]);
			}
			printf("\n");
		
		}		
		*/
		
		
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		
		/// ***********************************************************
        /// **** STEP 2: CreateBuckets
        /// **** STEP 2.2: Declare and Generate Pivots and Slopes
        /// **** Host generate KD intervals and send it to other ranks
        /// ***********************************************************
        
        
        // Find bucket sizes using a randomized selection
        if (rank == 0) {
        
		    generatePivots_B<T>(pivots, slopes, d_sampleVector_Host, sampleSize_host, numPivots, 
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
		    printf("%d\n", numUniqueBuckets);
		    for (int i = 0; i < numPivots - 1; i++) {
		    	printf("%d: %lf, %u, %u, %u\n", i, slopes[i], pivotsLeft[i],
		    									pivotsRight[i], kthnumBuckets[i]);
		    
		    }
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
		
		// cudaDeviceSynchronize();
		// MPI_Barrier(MPI_COMM_WORLD);
		
		
		/// ***********************************************************
        /// **** STEP 3: AssignBuckets
        /// **** Using the function assignSmartBucket
        /// ***********************************************************
        
        if (true) {
		    assignSmartBucket_distributive_CALL_B
		    		(d_vector_local, length_local, d_elementToBucket, d_slopes, 
					 d_pivotsLeft,  d_pivotsRight, d_kthnumBuckets, d_bucketCount, 
					 numUniqueBuckets, numBuckets, offset, numBlocks, threadsPerBlock);
        }
               									
        cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		/// ***********************************************************
        /// **** STEP 4: IdentifyActiveBuckets
        /// **** STEP 4.1 Update the bucketCount
        /// **** Each slots calculates bucketCount and
        /// **** send it to the host to cumulate
        /// ***********************************************************
        
        // each slot updates their bucketCount and send information to the host
        if (true) {
        	sumCounts_CALL_B(d_bucketCount, numBuckets, numBlocks, threadsPerBlock);
        	
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
        
        // cudaDeviceSynchronize();
		// MPI_Barrier(MPI_COMM_WORLD);
        
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
       
		
		/// ***********************************************************
        /// **** STEP 4: IdentifyActiveBuckets
        /// **** STEP 4.2 Find and update the kth buckets
        /// **** and their respective indices
        /// ***********************************************************
        
        if (rank == 0) {
        	findKBuckets_B(h_bucketCount_Host, numBuckets, kVals, numKs, 
        				 kthBucketScanner, kthBuckets, numBlocks);
        				 
        	updatekVals_distributive_B<T>(kVals, &numKs, output, kIndices, &length_host, 
        					&length_host_Old, h_bucketCount_Host, kthBuckets, kthBucketScanner,
                            reindexCounter, uniqueBuckets, uniqueBucketCounts, 
                            &numUniqueBuckets, &numUniqueBucketsOld); 
		  				 
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
		
		
		
		/// ***********************************************************
        /// **** STEP 4: IdentifyActiveBuckets
        /// **** STEP 4.3 Find and update kVals
        /// ***********************************************************
		
		// host send potential k-order statistics information to each slot
        if (rank == 0) {
        	// test part
        	// printf("Rank 0, tempKorderLength: %d\n", tempKorderLength);
        	for (int i = 1; i < RANK_NUM; i++) {
				MPI_Send_CALL(&numKs, 1, i, 12, MPI_COMM_WORLD);
        	}
        }
		
		// each slots receive information and perform updateOutput function
        if (rank != 0) {
			MPI_Recv_CALL(&numKs, 1, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
			
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		
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
    		
    		updateReindexCounter_distributive_B
    					(reindexCounter, h_bucketCount, uniqueBuckets, &length_local,
    					 &length_local_Old, numUniqueBuckets);
    		
    		cudaMemcpy(d_reindexCounter, reindexCounter, numUniqueBuckets * sizeof(unsigned int), 
    				   cudaMemcpyHostToDevice);
    		cudaMemcpy(d_uniqueBuckets, uniqueBuckets, numUniqueBuckets * sizeof(unsigned int), 
    				   cudaMemcpyHostToDevice);
    				   
    		reindexCounts_CALL_B(d_bucketCount, numBuckets, numBlocks, d_reindexCounter, 
    						   d_uniqueBuckets, numUniqueBuckets, threadsPerBlock);
    	}
    	
    	cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		/// ***********************************************************
        /// **** Step 5.1.2: copy active elements
        /// ***********************************************************
    
    	// each slots copy the active elements
    	if (true) {
    		copyElements_distributive_CALL_B
    			(d_vector_local, d_newvector, length_local_Old, d_elementToBucket, 
    			 d_uniqueBuckets, numUniqueBuckets, d_bucketCount, 
    			 numBuckets, offset, threadsPerBlock, numBlocks);
    			 
    		swapPointers_B(&d_vector_local, &d_newvector);
    	}
    
    	cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		
		/// ***********************************************************
        /// **** Step 5.1.3: send active elements to host
        /// ***********************************************************
		
		
		T* newvector = (T*)malloc(length_local * sizeof(T));
		cudaMemcpy(newvector, d_vector_local, length_local * sizeof(T),
				   cudaMemcpyDeviceToHost);
				   
		// each slot send active vector to host
		if (rank != 0) {
			MPI_Send_CALL(&length_local, 1, 0, 21, MPI_COMM_WORLD);
			MPI_Send_CALL(newvector, length_local, 0, 22, MPI_COMM_WORLD);
		}
		
		// cudaDeviceSynchronize();
		// MPI_Barrier(MPI_COMM_WORLD);
		
		T* newvector_host = NULL;
		T* newvector_receive = NULL;
		int length_temp;
		int index = 0;
		// host receive each vector and copy it to vector_host
		if (rank == 0) {
			newvector_host = (T*)malloc(length_host * sizeof(T));
			newvector_receive = (T*)malloc(length_host * sizeof(T));
			for (int j = 0; j < length_local; j++) {
				newvector_host[index] = newvector[j];
				index++;
			}
			
			for (int i = 1; i < RANK_NUM; i++) {
				MPI_Recv_CALL(&length_temp, 1, i, 21, MPI_COMM_WORLD,
							  MPI_STATUS_IGNORE);
				MPI_Recv_CALL(newvector_receive, length_temp, i, 22, 
							  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (int j = 0; j < length_temp; j++) {
					newvector_host[index] = newvector_receive[j];
					index++;
				}
			
			}
		}
		
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		
		/*
		// test part
		for (int i = 0; i < RANK_NUM; i++) {
			if (rank == i) {
				printf("%d: %d\n", rank, length_local);
			}	
		}
		
		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		
		if (rank == 0) {
			printf("host length: %d\n", length_host);
		}
		*/
		
		
	    /// ***********************************************************
		/// **** STEP 6: sort&choose
		/// **** Using thrust::sort on the reduced vector and the
		/// **** updated indices of the order statistics, 
		/// **** we solve the reduced problem.
		/// ***********************************************************
		
		
		if (rank == 0) {
			cudaFree(d_newvector);
		    cudaMalloc(&d_newvector, length_host * sizeof(T));
		    cudaMemcpy(d_newvector, newvector_host, length_host * sizeof(T),
		    		   cudaMemcpyHostToDevice);
		    		   
		    // sort the vector
		    sort_vector_CALL_B(d_newvector, length_host);
		    
		    T* d_output = (T*)d_elementToBucket;
		    cudaMemcpy (d_kVals, kVals, numKs * sizeof (uint), 
		                      cudaMemcpyHostToDevice);
			cudaMemcpy (d_kIndices, kIndices, numKs * sizeof (uint), 
				                  cudaMemcpyHostToDevice);
				                  
			copyValuesInChunk_CALL_B(d_output, d_newvector, d_kVals, 
					d_kIndices, numKs, numBlocks, threadsPerBlock);
					
			cudaMemcpy (output, d_output, 
					   (numKs) * sizeof (T), 
		                      cudaMemcpyDeviceToHost);
        }
        
        
        MPI_Barrier(MPI_COMM_WORLD);
		cudaDeviceSynchronize();
		
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
        free(sampleVector);


        cudaFree(d_slopes);
        cudaFree(d_pivots);
        cudaFree(d_pivotsLeft);
        cudaFree(d_pivotsRight);
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
        cudaFree(d_sampleVector);
        
        if (rank == 0) {
        	free(h_bucketCount_Host);
        	free(h_bucketCount_Receive);
        	free(sampleVector_Host);
        	cudaFree(d_sampleVector_Host);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
		cudaDeviceSynchronize();
		
		
		
		return 0;
		
		
		
		
	}
	
	template int distributedBucketMultiselect
			(int* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, int* output, int blocks, int threads, int numBuckets, int numPivots, 
		     int rank);
	template unsigned int distributedBucketMultiselect
			(unsigned int* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, unsigned int* output, int blocks, int threads, int numBuckets, int numPivots, 
		     int rank);
	template float distributedBucketMultiselect
			(float* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, float* output, int blocks, int threads, int numBuckets, int numPivots, 
		     int rank);
	template double distributedBucketMultiselect
			(double* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, double* output, int blocks, int threads, int numBuckets, int numPivots, 
		     int rank);



	template <typename T>
	T distributedBucketMultiselectWrapper (T* d_vector_local, int length_local, 
			unsigned int * kVals_ori, int numKs, T* output, int blocks, int threads, 
			int rank) {
			
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
		
		/*
		if (rank == 0) {
			T* vector_local = (T*)malloc(length_local * sizeof(T));
			cudaMemcpy(vector_local, d_vector_local, length_local * sizeof(T),
					   cudaMemcpyDeviceToHost);
			for (int i = 0; i < 200; i++) {
				printf("%u  ", vector_local[i]);
			
			}
			printf("\n");
		
		}
		*/
		

		distributedBucketMultiselect(d_vector_local, length_local, length_total, 
			kVals, numKs, output, blocks, threads, numBuckets, 17, rank);
			
		

		MPI_Barrier(MPI_COMM_WORLD);
		cudaDeviceSynchronize();

		free(kVals);
		
		MPI_Barrier(MPI_COMM_WORLD);
		cudaDeviceSynchronize();

		return 1;
	}
	
	template int distributedBucketMultiselectWrapper 
			(int* d_vector_local, int length_local, unsigned int * kVals_ori, int numKs,
             int* output, int blocks, int threads, int rank);
    template unsigned int distributedBucketMultiselectWrapper 
			(unsigned int* d_vector_local, int length_local, unsigned int * kVals_ori, int numKs,
             unsigned int* output, int blocks, int threads, int rank);
    template float distributedBucketMultiselectWrapper 
			(float* d_vector_local, int length_local, unsigned int * kVals_ori, int numKs,
             float* output, int blocks, int threads, int rank);
    template double distributedBucketMultiselectWrapper 
			(double* d_vector_local, int length_local, unsigned int * kVals_ori, int numKs,
             double* output, int blocks, int threads, int rank);

}
		
		

/// ***********************************************************
/// ***********************************************************
/// **** distributedSMOS: the main function
/// ***********************************************************
/// ***********************************************************
/*
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



    //	MPI_Barrier(MPI_COMM_WORLD);
    
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
			kVals[i] = i * 7000 + 1;
			output[i] = 0;
		}

    }

    MPI_Barrier(MPI_COMM_WORLD);
	
    
    
	DistributedBucketMultiselect::distributedBucketMultiselect
			(d_vector_local, length_local, length_local * RANK_NUM, kVals, numKs, output, 
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
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		



