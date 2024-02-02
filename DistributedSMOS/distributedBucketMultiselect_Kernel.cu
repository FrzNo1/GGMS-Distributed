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

#include "distributedBucketMultiselect_Kernel.cuh"

#define MAX_THREADS_PER_BLOCK 1024

using namespace std;

// thrust::minmax_element function
template <typename T>
void minmax_element_CALL_B(T* d_vector, int length, T* maximum, T* minimum) {
	thrust::device_ptr<T>dev_ptr(d_vector);
	thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result =
                thrust::minmax_element(dev_ptr, dev_ptr + length);
	
	*minimum = *result.first;
	*maximum = *result.second;
}



template void minmax_element_CALL_B(int* d_vector, int length, int* maximum, int* minimum);
template void minmax_element_CALL_B(unsigned int* d_vector, int length, unsigned int* maximum, unsigned int* minimum);
template void minmax_element_CALL_B(double* d_vector, int length, double* maximum, double* minimum);
template void minmax_element_CALL_B(float* d_vector, int length, float* maximum, float* minimum);


// thrust::sort_by_key function
void sort_by_key_CALL_B(unsigned int* d_kVals, unsigned int* d_kIndices, int numKs) {
	thrust::device_ptr<unsigned int>kVals_ptr(d_kVals);
	thrust::device_ptr<unsigned int>kIndices_ptr(d_kIndices);
	thrust::sort_by_key(kVals_ptr, kVals_ptr + numKs, kIndices_ptr);
}

// thrust::sort_by_key function
template <typename T>
void sort_vector_CALL_B(T* d_input, int length) {
	thrust::device_ptr<T>input_ptr(d_input);
	thrust::sort(input_ptr, input_ptr + length);
}

template void sort_vector_CALL_B(int* d_input, int length);
template void sort_vector_CALL_B(unsigned int* d_input, int length);
template void sort_vector_CALL_B(float* d_input, int length);
template void sort_vector_CALL_B(double* d_input, int length);


/// ***********************************************************
/// ***********************************************************
/// **** HELPER CPU FUNCTIONS
/// ***********************************************************
/// ***********************************************************

/* This function initializes a vector to all zeros on the host (CPU).
 */
template<typename T>
void setToAllZero_B (T * d_vector, int length) {
    cudaMemset(d_vector, 0, length * sizeof(T));
}


/* This function finds the bin containing the kth element we are looking for (works on
   the host). While doing the scan, it stores the sum-so-far of the number of elements in
   the buckets where k values fall into.

   markedBuckets : buckets containing the corresponding k values
   sums : sum-so-far of the number of elements in the buckets where k values fall into
*/
int findKBuckets_B(unsigned int * h_bucketCount, int numBuckets, 
				const unsigned int * kVals, int numKs, unsigned int * sums, 
				unsigned int * markedBuckets, int numBlocks) {
    int kBucket = 0;
    int k;
    int sum = h_bucketCount[0];

    for(int i = 0; i < numKs; i++) {
        k = kVals[i];
        while ((sum < k) & (kBucket < numBuckets - 1)) {
            kBucket++;
            sum += h_bucketCount[kBucket];
        }
        markedBuckets[i] = kBucket;
        sums[i] = sum - h_bucketCount[kBucket];
    }

    return 0;
}


/*
 * This function updates the correct kth orderstats if the bin only contains one element. While going through the
 * list of orderstats, it updates K since we have reduced the problem size to elements in the kth bucket. In
 * addition, it updates the unique buckets list to avoid the situation where two order share the same buckets.
 *
 * kthBucketScanner:  sum-so-far of the number of elements in the buckets where k values fall into
 * uniqueBuckets:  the list to store all buckets which are active with no repeats
 * tempKorderBucket:  buckets which have only one element. That is, the bucket with correct kth orderstats
 */
template <typename T>
int updatekVals_distributive_B
			(unsigned int * kVals, int * numKs, T * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld) {
    int index = 0;
    int numKsindex = 0;
    *numUniqueBucketsOld = *numUniqueBuckets;
    *numUniqueBuckets = 0;
    *lengthOld = *length;

	/*
    // go through the markedbucket list. If there is only one element in array, we update it to tempKorderBucket
    while (index < *numKs) {
        if (h_bucketCount[markedBuckets[index]] == 1) {
            tempKorderIndeces[*tempKorderLength] = kIndicies[index];
            tempKorderBucket[*tempKorderLength] = markedBuckets[index];
            (*tempKorderLength)++;
            index++;
            continue;
        }

        break;
    }
    */

    // get the index of the first buckets with more than one elements in it
    // add the number of elements and updates correct kth order
    // if (index < *numKs) {
        uniqueBuckets[0] = markedBuckets[index];
        uniqueBucketCounts[0] = h_bucketCount[markedBuckets[index]];
        reindexCounter[0] = 0;
        *numUniqueBuckets = 1;
        kVals[0] = kVals[index] - kthBucketScanner[index];
        kIndicies[0] = kIndicies[index];
        numKsindex++;
        index++;
    // }

    // go through the markedbuckets list. If there is only one element in that bucket, updates it to
    // tempKorderBucket; if there is more than one, updates it to uniqueBucket
    for ( ; index < *numKs; index++) {

		/*
        // case if there is only one element
        if (h_bucketCount[markedBuckets[index]] == 1) {
            tempKorderIndeces[*tempKorderLength] = kIndicies[index];
            tempKorderBucket[*tempKorderLength] = markedBuckets[index];
            (*tempKorderLength)++;
            continue;
        }
        */

        // case if the there is more than one element in the bucket and the bucket is not repeat with last one
        if (markedBuckets[index] != uniqueBuckets[(*numUniqueBuckets) - 1]) {
            uniqueBuckets[*numUniqueBuckets] = markedBuckets[index];
            uniqueBucketCounts[*numUniqueBuckets] = h_bucketCount[markedBuckets[index]];
            reindexCounter[*numUniqueBuckets] = reindexCounter[(*numUniqueBuckets) - 1]
                                                + uniqueBucketCounts[(*numUniqueBuckets) - 1];
            (*numUniqueBuckets)++;
        }

        // update korder
        kVals[numKsindex] = reindexCounter[(*numUniqueBuckets) - 1] + kVals[index] - kthBucketScanner[index];
        kIndicies[numKsindex] = kIndicies[index];
        numKsindex++;
    }

    // update numKs and length of vector
    *numKs = numKsindex;
    if (*numKs > 0)
        *length = reindexCounter[(*numUniqueBuckets) - 1] + uniqueBucketCounts[(*numUniqueBuckets) - 1];


    return 0;
}

template int updatekVals_distributive_B
			(unsigned int * kVals, int * numKs, int * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld);
template int updatekVals_distributive_B
			(unsigned int * kVals, int * numKs, unsigned int * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld);
template int updatekVals_distributive_B
			(unsigned int * kVals, int * numKs, float * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld);
template int updatekVals_distributive_B
			(unsigned int * kVals, int * numKs, double * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld);
             
             
/*
 * Documentation
 * 
 */             
int updateReindexCounter_distributive_B
			(unsigned int* reindexCounter, unsigned int* h_bucketCount, unsigned int* uniqueBuckets,
			 int* length, int* length_Old, int numUniqueBuckets) {
	reindexCounter[0] = 0;
	*length_Old = *length;
	
	for (int i = 1; i < numUniqueBuckets; i++) {
		reindexCounter[i] = reindexCounter[i - 1] + h_bucketCount[uniqueBuckets[i - 1]];
	}
	
	*length = reindexCounter[numUniqueBuckets - 1] + 
			  h_bucketCount[uniqueBuckets[numUniqueBuckets - 1]];			 
	
	return 0;	 
}

/*
 * This function swap pointers for the two lists
 */
template <typename T>
void swapPointers_B(T** a, T** b) {
    T * temp = * a;
    * a = * b;
    * b = temp;
}

template void swapPointers_B(int** a, int** b);
template void swapPointers_B(unsigned int** a, unsigned int** b);
template void swapPointers_B(float** a, float** b);
template void swapPointers_B(double** a, double** b);

/// ***********************************************************
/// ***********************************************************
/// **** HELPER GPU FUNCTIONS-KERNELS
/// ***********************************************************
/// ***********************************************************

/*
 * Documentation
 */
template <typename T>
__global__ void generateSamples_distributive_B
					(T* d_vector, T* d_sampleVector, int length_local, int sampleSize_local) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = length_local / sampleSize_local;
	
	if (index < sampleSize_local)
		d_sampleVector[index] = d_vector[index * offset];	
}



		
/* This function assigns elements to buckets based on the pivots and slopes determined
   by a randomized sampling of the elements in the vector. At the same time, this
   function keeps track of count.

   d_elementToBucket : bucket assignment for every array element
   d_bucketCount : number of element that falls into the indexed buckets within the block
*/
template <typename T>
__global__ void assignSmartBucket_distributive_B
				(T * d_vector, int length, unsigned int * d_elementToBucket,
                 double * slopes, T * pivotsLeft, T * pivotsRight,
                 unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
                 int numUniqueBuckets, int numBuckets, int offset) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int bucketIndex;
    int threadIndex = threadIdx.x;

    /*
    if (index == 1) {
        printf("assignmentSmartBucket in kernel\n\n");
    }
     */


    //variables in shared memory for fast access
    extern __shared__ unsigned int array[];
    double * sharedSlopes = (double *)array;
    T * sharedPivotsLeft = (T *)&sharedSlopes[numUniqueBuckets];
    unsigned int * sharedkthNumBuckets = (unsigned int *)&sharedPivotsLeft[numUniqueBuckets];
    unsigned int * sharedBuckets = (unsigned int *)&sharedkthNumBuckets[numUniqueBuckets];

    /*
    if (index < 1) {
        printf("executed to phase 0\n\n");
        /*
        for (int bb=0; bb < 16; bb++) {
            printf("bb=%d, vec=%d, elemtobuck=%d, slopes=%lf buckCout=%d, pleft=%d, pright=%d \n ", 
            	   bb, d_vector[bb], d_elementToBucket[bb],slopes[bb], 
            	   d_bucketCount[bb], pivotsLeft[bb], pivotsRight[bb]);
        }

        printf("\n \n \n");

    }
    */

    //reading bucket counts into shared memory where increments will be performed
    for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) {

        if (threadIndex < numBuckets)
            sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex] = 0;
    }

    /*
    //    if (index < length) {
    if (index < 1)
        printf("executed to phase 1\n\n");
        */


    if (threadIndex < numUniqueBuckets) {
        sharedPivotsLeft[threadIndex] = pivotsLeft[threadIndex];
        sharedSlopes[threadIndex] = slopes[threadIndex];
        sharedkthNumBuckets[threadIndex] = kthNumBuckets[threadIndex];
        //printf("PL=%d, Slps=%lf, kNumB=%d \n", pivotsLeft[threadIndex], slopes[threadIndex], kthNumBuckets[threadIndex]);
        //printf("sPL=%d, sSlps=%lf, skNumB=%d \n", sharedPivotsLeft[threadIndex], sharedSlopes[threadIndex], sharedkthNumBuckets[threadIndex]);
    }

    /*
    if (index < 1)
        printf("executed to phase 2\n\n");

    //       if (index < length)
    //         printf("index=%d, length=%d, numUniqueBuckets=%d, offset=%d \n", index, length, numUniqueBuckets, offset);
     */

    __syncthreads();

    /*
    if (index < 1)
        printf("executed to phase 3\n\n");
        */


    //assigning elements to buckets and incrementing the bucket counts
    if (index < length) {
        for (int i = index; i < length; i += offset) {
            T num = d_vector[i];
            int minPivotIndex = 0;
            int maxPivotIndex = numUniqueBuckets;
            int midPivotIndex;

            // find the index of left pivots that is greatest s.t. lower than or equal to
            // num using binary search
            for (int j = 1; j < numUniqueBuckets; j *= 2) {
                midPivotIndex = (maxPivotIndex + minPivotIndex) / 2;
                if (num >= pivotsLeft[midPivotIndex])
                    minPivotIndex = midPivotIndex;
                else
                    maxPivotIndex = midPivotIndex;
            }


            bucketIndex = sharedkthNumBuckets[minPivotIndex]
                          + (unsigned int) (((double)num - (double)sharedPivotsLeft[minPivotIndex])
                                   * sharedSlopes[minPivotIndex]);


            // potential to remove the for loop
            if (sharedPivotsLeft[minPivotIndex] != pivotsRight[minPivotIndex]) {
                if (bucketIndex >= numBuckets) {
                    bucketIndex = numBuckets - 1;
                }
                else if (minPivotIndex < numUniqueBuckets - 1) {
                    if (bucketIndex >= sharedkthNumBuckets[minPivotIndex + 1]) {
                        bucketIndex = sharedkthNumBuckets[minPivotIndex + 1] - 1;
                    }
                }
            }

            d_elementToBucket[i] = bucketIndex;
            atomicInc(sharedBuckets + bucketIndex, length);

            //             printf("%d, %d;  ", d_vector[i], d_elementToBucket[i]);
        }
    }

    /*
    if (index < 1)
        printf("executed to phase 4\n\n");
        */


    //    } // closes the if (index < max(length))

    __syncthreads();

    //reading bucket counts from shared memory back to global memory
    for (int i = 0; i <(numBuckets / MAX_THREADS_PER_BLOCK); i++)
        if (threadIndex < numBuckets)
            *(d_bucketCount + blockIdx.x * numBuckets
              + i * MAX_THREADS_PER_BLOCK + threadIndex) =
                    *(sharedBuckets + i * MAX_THREADS_PER_BLOCK + threadIndex);

    /*
    if (index < 1)
        printf("executed to phase 5\n\n");
        */

} // closes the kernel
		
		
/* This function cumulatively sums the count of every block for a given bucket s.t. the
   last block index holds the total number of elements falling into that bucket all over the
   array.
   updates d_bucketCount
*/
__global__ void sumCounts_B(unsigned int * d_bucketCount, const int numBuckets
        , const int numBlocks) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int j=1; j<numBlocks; j++)
        d_bucketCount[index + numBuckets*j] += d_bucketCount[index + numBuckets*(j-1)];
}


/* This function reindexes the buckets counts for every block according to the
   accumulated d_reindexCounter counter for the reduced vector.
   updates d_bucketCount
*/
__global__ void reindexCounts_B(unsigned int * d_bucketCount, int numBuckets, int numBlocks,
                              unsigned int * d_reindexCounter, unsigned int * d_uniqueBuckets,
                              const int numUniqueBuckets) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIndex < numUniqueBuckets) {
        int index = d_uniqueBuckets[threadIndex];
        unsigned int add = d_reindexCounter[threadIndex];

        for (int j = 0; j < numBlocks; j++)
            d_bucketCount[index + numBuckets * j] += add;
    }
}



/* This function copies the elements of buckets that contain kVals into a newly allocated
   reduced vector space.
   newArray - reduced size vector containing the essential elements
*/
template <typename T>
__global__ void copyElements_distributive_B 
					(T * d_vector, T * d_newvector, int lengthOld, 
					 unsigned int * elementToBuckets, unsigned int * uniqueBuckets, 
					 int numUniqueBuckets, unsigned int * d_bucketCount, 
					 int numBuckets, unsigned int offset) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex;
    int loop = numBuckets / MAX_THREADS_PER_BLOCK;

    extern __shared__ unsigned int sharedBuckets[];

    for (int i = 0; i <= loop; i++) {
        threadIndex = i * blockDim.x + threadIdx.x;
        if (threadIndex < numUniqueBuckets)
            sharedBuckets[threadIndex] = uniqueBuckets[threadIndex];
    }

    __syncthreads();


    if (index < lengthOld) {

        for (int i = index; i < lengthOld; i += offset) {
            unsigned int temp = elementToBuckets[i];


            /*
            for (int j = 0; j < numUniqueBuckets; j++) {
                if (temp == sharedBuckets[j]) {
                    d_newvector[atomicDec(d_bucketCount + blockIdx.x * numBuckets
                                          + sharedBuckets[j], lengthOld) - 1] = d_vector[i];
                    break;
                }
            }
            */



            int minBucketIndex = 0;
            int maxBucketIndex = numUniqueBuckets - 1;
            int midBucketIndex;

            for (int j = 1; j < numUniqueBuckets; j *= 2) {
                midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;
                if (temp > sharedBuckets[midBucketIndex])
                    minBucketIndex = midBucketIndex + 1;
                else
                    maxBucketIndex = midBucketIndex;
            }

            if (temp == sharedBuckets[maxBucketIndex])
                d_newvector[atomicDec(d_bucketCount + blockIdx.x * numBuckets
                                      + sharedBuckets[maxBucketIndex], lengthOld) - 1] = d_vector[i];

        }
    }

    // needs to swap d_vector with d_newvector
}

/* This function speeds up the copying process the requested kVals by clustering them
 together.
*/
template <typename T>
__global__ void copyValuesInChunk_B (T * outputVector, T * inputVector, uint * kList
                                 , uint * kIndices, int kListCount) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int loop = kListCount / MAX_THREADS_PER_BLOCK;

	for (int i = 0; i <= loop; i++) {      
	  if (idx < kListCount)
		*(outputVector + *(kIndices + idx)) = *(inputVector + *(kList + idx) - 1);
	}
}


/// ***********************************************************
/// ***********************************************************
/// **** HELPER GPU FUNCTIONS LIBRARIES
/// ***********************************************************
/// ***********************************************************
template <typename T>
void generateSamples_distributive_CALL_B
			(T* d_vector, T* d_sampleVector, int length_local, int sampleSize_local) {
	 generateSamples_distributive_B<<<8, 1024>>> 
	 		(d_vector, d_sampleVector, length_local, sampleSize_local);		
			}
template void generateSamples_distributive_CALL_B
			(int* d_vector, int* d_sampleVector, 
			 int length_local, int sampleSize_local);
template void generateSamples_distributive_CALL_B
			(unsigned int* d_vector, unsigned int* d_sampleVector, 
			 int length_local, int sampleSize_local);
template void generateSamples_distributive_CALL_B
			(float* d_vector, float* d_sampleVector, 
			 int length_local, int sampleSize_local);
template void generateSamples_distributive_CALL_B
			(double* d_vector, double* d_sampleVector, 
			 int length_local, int sampleSize_local);
			 
			 
			 
			 
			 
template <typename T>
void assignSmartBucket_distributive_CALL_B
			(T * d_vector, int length, unsigned int * d_elementToBucket,
             double * slopes, T * pivotsLeft, T * pivotsRight,
             unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
             int numUniqueBuckets, int numBuckets, int offset, 
             int numBlocks, int threadsPerBlock) {
                                         
	int sharedMemorySize = numUniqueBuckets * sizeof(T) + 
						   numUniqueBuckets * sizeof(double) + 
                           numUniqueBuckets * sizeof(unsigned int) + 
                           numBuckets * sizeof(unsigned int);
                                         
    assignSmartBucket_distributive_B<T><<<numBlocks, threadsPerBlock, sharedMemorySize>>>
        		(d_vector, length, d_elementToBucket, slopes, pivotsLeft, pivotsRight,
                 kthNumBuckets, d_bucketCount, numUniqueBuckets, numBuckets, offset);                                
}

template void assignSmartBucket_distributive_CALL_B
			(int * d_vector, int length, unsigned int * d_elementToBucket,
		     double * slopes, int * pivotsLeft, int * pivotsRight,
		     unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
		     int numUniqueBuckets, int numBuckets, int offset,
		     int numBlocks, int threadsPerBlock);
template void assignSmartBucket_distributive_CALL_B
			(unsigned * d_vector, int length, unsigned int * d_elementToBucket,
        	 double * slopes, unsigned int * pivotsLeft, unsigned int * pivotsRight,
        	 unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
        	 int numUniqueBuckets, int numBuckets, int offset,
          	 int numBlocks, int threadsPerBlock);
template void assignSmartBucket_distributive_CALL_B
			(float * d_vector, int length, unsigned int * d_elementToBucket,
        	 double * slopes, float * pivotsLeft, float * pivotsRight,
        	 unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
        	 int numUniqueBuckets, int numBuckets, int offset,
        	 int numBlocks, int threadsPerBlock);
template void assignSmartBucket_distributive_CALL_B
			(double * d_vector, int length, unsigned int * d_elementToBucket,
        	 double * slopes, double * pivotsLeft, double * pivotsRight,
        	 unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
        	 int numUniqueBuckets, int numBuckets, int offset,
        	 int numBlocks, int threadsPerBlock);
			 
			 
void sumCounts_CALL_B(unsigned int * d_bucketCount, const int numBuckets, 
					const int numBlocks, int threadsPerBlock) {
	sumCounts_B<<<numBuckets/threadsPerBlock, threadsPerBlock>>>
		(d_bucketCount, numBuckets, numBlocks);
}
			 
void reindexCounts_CALL_B(unsigned int * d_bucketCount, int numBuckets, int numBlocks,
                        unsigned int * d_reindexCounter, unsigned int * d_uniqueBuckets,
                        const int numUniqueBuckets, int threadsPerBlock) {
	reindexCounts_B<<<(int) ceil((float)numUniqueBuckets/threadsPerBlock), threadsPerBlock>>>
          (d_bucketCount, numBuckets, numBlocks, d_reindexCounter, d_uniqueBuckets, 
           numUniqueBuckets);  
                        
}		 
			 
			 
template <typename T>
void copyElements_distributive_CALL_B
			(T * d_vector, T * d_newvector, int lengthOld, 
			 unsigned int * elementToBuckets, unsigned int * uniqueBuckets, 
			 int numUniqueBuckets, unsigned int * d_bucketCount, 
			 int numBuckets, unsigned int offset, int threadsPerBlock,
			 int numBlocks) {
	copyElements_distributive_B<T><<<numBlocks, threadsPerBlock, 
								   numUniqueBuckets * sizeof(unsigned int)>>>
			(d_vector, d_newvector, lengthOld, elementToBuckets, uniqueBuckets, 
			 numUniqueBuckets, d_bucketCount, numBuckets, offset);
}

template void copyElements_distributive_CALL_B
			(int * d_vector, int * d_newvector, int lengthOld, 
			 unsigned int * elementToBuckets, unsigned int * uniqueBuckets, 
			 int numUniqueBuckets, unsigned int * d_bucketCount, 
			 int numBuckets, unsigned int offset, int threadsPerBlock,
			 int numBlocks);
template void copyElements_distributive_CALL_B
			(unsigned int * d_vector, unsigned int * d_newvector, int lengthOld, 
			 unsigned int * elementToBuckets, unsigned int * uniqueBuckets, 
			 int numUniqueBuckets, unsigned int * d_bucketCount, 
			 int numBuckets, unsigned int offset, int threadsPerBlock,
			 int numBlocks);
template void copyElements_distributive_CALL_B
			(float * d_vector, float * d_newvector, int lengthOld, 
			 unsigned int * elementToBuckets, unsigned int * uniqueBuckets, 
			 int numUniqueBuckets, unsigned int * d_bucketCount, 
			 int numBuckets, unsigned int offset, int threadsPerBlock,
			 int numBlocks);
template void copyElements_distributive_CALL_B
			(double * d_vector, double * d_newvector, int lengthOld, 
			 unsigned int * elementToBuckets, unsigned int * uniqueBuckets, 
			 int numUniqueBuckets, unsigned int * d_bucketCount, 
			 int numBuckets, unsigned int offset, int threadsPerBlock,
			 int numBlocks);
			 
			 
template <typename T>
void copyValuesInChunk_CALL_B(T * outputVector, T * inputVector, uint * kList, 
                            uint * kIndices, int kListCount, int numBlocks, 
                            int threadsPerBlock) {
	copyValuesInChunk_B<T><<<numBlocks, threadsPerBlock>>>
			(outputVector, inputVector, kList, kIndices, kListCount);
			             
}

template void copyValuesInChunk_CALL_B
		(int* outputVector, int* inputVector, uint * kList, 
         uint * kIndices, int kListCount, int numBlocks, 
         int threadsPerBlock);
template void copyValuesInChunk_CALL_B
		(unsigned int* outputVector, unsigned int* inputVector, uint * kList, 
         uint * kIndices, int kListCount, int numBlocks, 
         int threadsPerBlock);
template void copyValuesInChunk_CALL_B
		(float* outputVector, float* inputVector, uint * kList, 
         uint * kIndices, int kListCount, int numBlocks, 
         int threadsPerBlock);
template void copyValuesInChunk_CALL_B
		(double* outputVector, double* inputVector, uint * kList, 
         uint * kIndices, int kListCount, int numBlocks, 
         int threadsPerBlock);





		 
			 
			 
			 
			 
/// ***********************************************************
/// ***********************************************************
/// **** GENERATE KD PIVOTS
/// ***********************************************************
/// ***********************************************************

/* Hash function using Monte Carlo method
 */
__host__ __device__
unsigned int myhash_B(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}



/* RandomNumberFunctor
 */
struct RandomNumberFunctor_B :
        public thrust::unary_function<unsigned int, float> {
    unsigned int mainSeed;

    RandomNumberFunctor_B(unsigned int _mainSeed) :
            mainSeed(_mainSeed) {}

    __host__ __device__
    float operator()(unsigned int threadIdx)
    {
        unsigned int seed = myhash_B(threadIdx) * mainSeed;

        thrust::default_random_engine rng(seed);
        rng.discard(threadIdx);
        thrust::uniform_real_distribution<float> u(0, 1);

        return u(rng);
    }
};



/* This function creates a random vector of 1024 elements in the range [0 1]
 */
template <typename T>
void createRandomVector_B(T * d_vec, int size) {
    timeval t1;
    unsigned int seed;

    gettimeofday(&t1, NULL);
    seed = t1.tv_usec * t1.tv_sec;
    // seed = 1000000000;

    thrust::device_ptr<T> d_ptr(d_vec);
    thrust::transform (thrust::counting_iterator<unsigned int>(0),
                       thrust::counting_iterator<unsigned int>(size),
                       d_ptr, RandomNumberFunctor_B(seed));
}



/* This function maps the [0 1] range to the [0 vectorSize] and
   grabs the corresponding elements.
*/
template <typename T>
__global__ void enlargeIndexAndGetElements_B (T * in, T * list, int size) {
    *(in + blockIdx.x*blockDim.x + threadIdx.x) =
            *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
}

__global__ void enlargeIndexAndGetElements_B (float * in, int * out, int * list, int size) {
    *(out + blockIdx.x * blockDim.x + threadIdx.x) =
            (int) *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
}


__global__ void enlargeIndexAndGetElements_B (float * in, unsigned int * out, unsigned int * list, int size) {
    *(out + blockIdx.x * blockDim.x + threadIdx.x) =
            (unsigned int) *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
}



/* This function generates Pivots from the random sampled data and calculates slopes.

   pivots - arrays of pivots
   slopes - array of slopes
*/
template <typename T>
void generatePivots_B (int * pivots, double * slopes, int * d_list, int sizeOfVector
        , int numPivots, int sizeOfSample, int totalSmallBuckets, int min, int max) {

    float * d_randomFloats;
    int * d_randomInts;
    int endOffset = 22;
    int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
    int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

    cudaMalloc (&d_randomFloats, sizeof (float) * sizeOfSample);

    d_randomInts = (int *) d_randomFloats;

    createRandomVector_B (d_randomFloats, sizeOfSample);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements_B<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
    , MAX_THREADS_PER_BLOCK>>>(d_randomFloats, d_randomInts, d_list,
                               sizeOfVector);



    pivots[0] = min;
    pivots[numPivots-1] = max;

    thrust::device_ptr<T>randoms_ptr(d_randomInts);
    thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

    cudaThreadSynchronize();

    // set the pivots which are next to the min and max pivots using the random element
    // endOffset away from the ends
    cudaMemcpy (pivots + 1, d_randomInts + endOffset - 1, sizeof (int)
            , cudaMemcpyDeviceToHost);
    cudaMemcpy (pivots + numPivots - 2, d_randomInts + sizeOfSample - endOffset - 1,
                sizeof (int), cudaMemcpyDeviceToHost);
    slopes[0] = numSmallBuckets / (double) (pivots[1] - pivots[0]);

    for (int i = 2; i < numPivots - 2; i++) {
        cudaMemcpy (pivots + i, d_randomInts + pivotOffset * (i - 1) + endOffset - 1,
                    sizeof (int), cudaMemcpyDeviceToHost);
        slopes[i - 1] = numSmallBuckets / (double) (pivots[i] - pivots[i - 1]);
    }

    // printf("\n\n\n\n%d %d %d %d %d\n\n\n\n", pivots[0], pivots[4], pivots[7], pivots[10], pivots[16]);

    slopes[numPivots - 3] = numSmallBuckets /
                            (double) (pivots[numPivots - 2] - pivots[numPivots - 3]);
    slopes[numPivots - 2] = numSmallBuckets /
                            (double) (pivots[numPivots - 1] - pivots[numPivots - 2]);

    cudaFree(d_randomFloats);
}

template void generatePivots_B<int>(int * pivots, double * slopes, int * d_list, int sizeOfVector, 
							 int numPivots, int sizeOfSample, int totalSmallBuckets, int min, int max);



/* This function generates Pivots from the random sampled data and calculates slopes.

   pivots - arrays of pivots
   slopes - array of slopes
*/
template <typename T>
void generatePivots_B (unsigned int * pivots, double * slopes, unsigned int * d_list, int sizeOfVector
        , int numPivots, int sizeOfSample, int totalSmallBuckets, unsigned int min, unsigned int max) {

    float * d_randomFloats;
    unsigned int * d_randomInts;
    int endOffset = 22;
    int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
    int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

    cudaMalloc (&d_randomFloats, sizeof (float) * sizeOfSample);

    d_randomInts = (unsigned int *) d_randomFloats;

    createRandomVector_B (d_randomFloats, sizeOfSample);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements_B<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
    , MAX_THREADS_PER_BLOCK>>>(d_randomFloats, d_randomInts, d_list,
                               sizeOfVector);



    pivots[0] = min;
    pivots[numPivots-1] = max;

    thrust::device_ptr<T>randoms_ptr(d_randomInts);
    thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

    cudaThreadSynchronize();

    // set the pivots which are next to the min and max pivots using the random element
    // endOffset away from the ends
    cudaMemcpy (pivots + 1, d_randomInts + endOffset - 1, sizeof (unsigned int)
            , cudaMemcpyDeviceToHost);
    cudaMemcpy (pivots + numPivots - 2, d_randomInts + sizeOfSample - endOffset - 1,
                sizeof (unsigned int), cudaMemcpyDeviceToHost);
    slopes[0] = numSmallBuckets / (double) (pivots[1] - pivots[0]);

    for (int i = 2; i < numPivots - 2; i++) {
        cudaMemcpy (pivots + i, d_randomInts + pivotOffset * (i - 1) + endOffset - 1,
                    sizeof (unsigned int), cudaMemcpyDeviceToHost);
        slopes[i - 1] = numSmallBuckets / (double) (pivots[i] - pivots[i - 1]);
    }

    // printf("\n\n\n\n%d %d %d %d %d\n\n\n\n", pivots[0], pivots[4], pivots[7], pivots[10], pivots[16]);

    slopes[numPivots - 3] = numSmallBuckets /
                            (double) (pivots[numPivots - 2] - pivots[numPivots - 3]);
    slopes[numPivots - 2] = numSmallBuckets /
                            (double) (pivots[numPivots - 1] - pivots[numPivots - 2]);

    cudaFree(d_randomFloats);
}

template void generatePivots_B<unsigned int>(unsigned int * pivots, double * slopes, unsigned int * d_list, int sizeOfVector, 
        				     int numPivots, int sizeOfSample, int totalSmallBuckets, unsigned int min, unsigned int max);

template <typename T>
void generatePivots_B (T * pivots, double * slopes, T * d_list, int sizeOfVector
        , int numPivots, int sizeOfSample, int totalSmallBuckets, T min, T max) {
    T * d_randoms;
    int endOffset = 22;
    int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
    int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

    cudaMalloc (&d_randoms, sizeof (T) * sizeOfSample);

    createRandomVector_B (d_randoms, sizeOfSample);

    // converts randoms floats into elements from necessary indices
    enlargeIndexAndGetElements_B<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
    , MAX_THREADS_PER_BLOCK>>>(d_randoms, d_list, sizeOfVector);

    pivots[0] = min;
    pivots[numPivots - 1] = max;

    thrust::device_ptr<T>randoms_ptr(d_randoms);
    thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

    cudaThreadSynchronize();

    // set the pivots which are endOffset away from the min and max pivots
    cudaMemcpy (pivots + 1, d_randoms + endOffset - 1, sizeof (T),
                cudaMemcpyDeviceToHost);
    cudaMemcpy (pivots + numPivots - 2, d_randoms + sizeOfSample - endOffset - 1,
                sizeof (T), cudaMemcpyDeviceToHost);
    slopes[0] = numSmallBuckets / ((double)pivots[1] - (double)pivots[0]);

    for (int i = 2; i < numPivots - 2; i++) {
        cudaMemcpy (pivots + i, d_randoms + pivotOffset * (i - 1) + endOffset - 1,
                    sizeof (T), cudaMemcpyDeviceToHost);
        slopes[i - 1] = numSmallBuckets / ((double) pivots[i] - (double) pivots[i - 1]);
    }

    slopes[numPivots - 3] = numSmallBuckets /
                            ((double)pivots[numPivots - 2] - (double)pivots[numPivots - 3]);
    slopes[numPivots - 2] = numSmallBuckets /
                            ((double)pivots[numPivots - 1] - (double)pivots[numPivots - 2]);

    cudaFree(d_randoms);
}


template void generatePivots_B(float * pivots, double * slopes, float * d_list, int sizeOfVector, 
							 int numPivots, int sizeOfSample, int totalSmallBuckets, float min, float max);
template void generatePivots_B(double * pivots, double * slopes, double * d_list, int sizeOfVector, 
							 int numPivots, int sizeOfSample, int totalSmallBuckets, double min, double max);
							 
							 
						 
			 
			 
			 
			 
			 
			 
			 
			 

