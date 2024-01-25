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

#include "iterativeSMOS.cuh"

namespace IterativeSMOS {
    using namespace std;


// #define SAFE
#define MAX_THREADS_PER_BLOCK 1024
#define CUTOFF 33554432
#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)

    /// ***********************************************************
    /// ***********************************************************
    /// **** HELPER SAFE FUNCTIONS
    /// ***********************************************************
    /// ***********************************************************




    /// ************** SAFETY ERROR CHECK FUNCTIONS *************


    void check_malloc_int(int *pointer, const char *message)
    {
        if ( pointer == NULL ) {
            printf("Malloc failed for %s.\n", message);
        }
    }

    void check_malloc_float(float *pointer, const char *message)
    {
        if ( pointer == NULL ) {
            printf("Malloc failed for %s.\n", message);
        }
    }

    void check_malloc_double(double *pointer, const char *message)
    {
        if ( pointer == NULL ) {
            printf("Malloc failed for %s.\n", message);
        }
    }

    void check_cudaMalloc(const char *message)
    {
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            fprintf(stderr, "Error: cudaMalloc failed for %s: %d\n", message, status);
        }
    }

    void Check_CUDA_Error(const char *message)
    {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "Error: %s: %s\n", message, cudaGetErrorString(error) );
            exit(-1);
        }
    }

    /*
    void Check_CUBLAS_Error(const char *message)
    {
        cublasStatus status = cublasGetError();
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf (stderr, "Error: %s: %d\n", message, status);
            exit(-1);
        }
    }
     */

    /*

    void Check_CURAND_Error(curandStatus_t curandCheck, const char *message)
    {
        if (curandCheck != CURAND_STATUS_SUCCESS) {
            fprintf (stderr, "Error: %s: %d\n", message, curandCheck);
            exit(-1);
        }
    }

     */



    void check_cudaMalloc2(cudaError_t status, const char *message)
    {
        if (status != cudaSuccess) {
            fprintf(stderr, "Error: cudaMalloc failed for %s: %d\n", message, status);
        }
    }




// ************** SAFETY ERROR CHECK WRAPPERS TO BE USED IN CODE *************



    inline void SAFEcudaMalloc2(cudaError_t status, const char *message)
    {
#ifdef SAFE
        check_cudaMalloc2(status, message);
#endif
    }



    inline void SAFEcudaMalloc(const char *message)
    {
#ifdef SAFE
        check_cudaMalloc(message);
#endif
    }

    inline void SAFEcuda(const char *message)
    {
#ifdef SAFE
        Check_CUDA_Error(message);
#endif
    }

    /*
    inline void SAFEcublas(const char *message)
    {
#ifdef SAFE
        Check_CUBLAS_Error(message);
#endif
    }
     */

    /*

    inline void SAFEcurand(curandStatus_t curandCheck, const char *message)
    {
#ifdef SAFE
        Check_CURAND_Error(curandCheck, message);
#endif
    }

     */

    inline void SAFEmalloc_int(int * pointer, const char *message)
    {
#ifdef SAFE
        check_malloc_int(pointer, message);
#endif
    }

    inline void SAFEmalloc_float(float *pointer, const char *message)
    {
#ifdef SAFE
        check_malloc_float(pointer, message);
#endif
    }

    inline void SAFEmalloc_double(double *pointer, const char *message)
    {
#ifdef SAFE
        check_malloc_double(pointer, message);
#endif
    }




    /// ***********************************************************
    /// ***********************************************************
    /// **** HELPER CPU FUNCTIONS
    /// ***********************************************************
    /// ***********************************************************

    /* This function initializes a vector to all zeros on the host (CPU).
     */
    template<typename T>
    void setToAllZero (T * d_vector, int length) {
        cudaMemset(d_vector, 0, length * sizeof(T));
    }


    /* This function finds the bin containing the kth element we are looking for (works on
       the host). While doing the scan, it stores the sum-so-far of the number of elements in
       the buckets where k values fall into.

       markedBuckets : buckets containing the corresponding k values
       sums : sum-so-far of the number of elements in the buckets where k values fall into
    */
    inline int findKBuckets
            (unsigned int * d_bucketCount, unsigned int * h_bucketCount, int numBuckets,
             const unsigned int * kVals, int numKs, unsigned int * sums, unsigned int * markedBuckets,
             int numBlocks) {
        // consider the last row which holds the total counts
        int sumsRowIndex= numBuckets * (numBlocks-1);

        cudaMemcpy(h_bucketCount, d_bucketCount + sumsRowIndex,
                   sizeof(unsigned int) * numBuckets, cudaMemcpyDeviceToHost);

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
     * This function goes through the list of orderstats, it updates K since we have reduced the problem size to
     * elements in the kth bucket. In addition, it updates the unique buckets list to avoid the situation where
     * two order share the same buckets.
     *
     * kthBucketScanner:  sum-so-far of the number of elements in the buckets where k values fall into
     * uniqueBuckets:  the list to store all buckets which are active with no repeats
     */
    template <typename T>
    inline int updatekVals
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

        // get the index of the first buckets with more than one elements in it
        // add the number of elements and updates correct kth order
        uniqueBuckets[0] = markedBuckets[index];
        uniqueBucketCounts[0] = h_bucketCount[markedBuckets[index]];
        reindexCounter[0] = 0;
        *numUniqueBuckets = 1;
        kVals[0] = kVals[index] - kthBucketScanner[index];
        kIndicies[0] = kIndicies[index];
        numKsindex++;
        index++;

        // go through the markedbuckets list. If there is more than one, updates it to uniqueBucket
        for ( ; index < *numKs; index++) {
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
    inline int updatekVals_iterative
            (unsigned int * kVals, int * numKs, T * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld,
             unsigned int * tempKorderBucket, unsigned int * tempKorderIndeces, int * tempKorderLength) {
        int index = 0;
        int numKsindex = 0;
        *numUniqueBucketsOld = *numUniqueBuckets;
        *numUniqueBuckets = 0;
        *lengthOld = *length;
        *tempKorderLength = 0;

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

        // get the index of the first buckets with more than one elements in it
        // add the number of elements and updates correct kth order
        if (index < *numKs) {
            uniqueBuckets[0] = markedBuckets[index];
            uniqueBucketCounts[0] = h_bucketCount[markedBuckets[index]];
            reindexCounter[0] = 0;
            *numUniqueBuckets = 1;
            kVals[0] = kVals[index] - kthBucketScanner[index];
            kIndicies[0] = kIndicies[index];
            numKsindex++;
            index++;
        }

        // go through the markedbuckets list. If there is only one element in that bucket, updates it to
        // tempKorderBucket; if there is more than one, updates it to uniqueBucket
        for ( ; index < *numKs; index++) {

            // case if there is only one element
            if (h_bucketCount[markedBuckets[index]] == 1) {
                tempKorderIndeces[*tempKorderLength] = kIndicies[index];
                tempKorderBucket[*tempKorderLength] = markedBuckets[index];
                (*tempKorderLength)++;
                continue;
            }

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

    /*
     * This function swap pointers for the two lists
     */
    template <typename T>
    void swapPointers(T** a, T** b) {
        T * temp = * a;
        * a = * b;
        * b = temp;
    }



    /// ***********************************************************
    /// ***********************************************************
    /// **** HELPER GPU FUNCTIONS-KERNELS
    /// ***********************************************************
    /// ***********************************************************


    /*
     * This function generate new buckets offset and slopes by giving the new pivots and number of elements in
     * that buckets
     *
     * pivotsLeft & pivotsRight:  the bounds of elements for each bucket
     * kthnumBuckets:  array to store bucket offset.
     */
    template <typename T>
    __global__ void generateBucketsandSlopes_iterative
            (T * pivotsLeft, T * pivotsRight, double * slopes,
             unsigned int * uniqueBucketsCounts, int numUniqueBuckets,
             unsigned int * kthnumBuckets, int length, int offset, int numBuckets) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        // Assign bucket number and slope to first to the second to last active buckets
        if (index < numUniqueBuckets - 1) {
            for (int i = index; i < numUniqueBuckets - 1; i += offset) {

                // assign bucket number
                kthnumBuckets[i] = max(uniqueBucketsCounts[i] * numBuckets / length, 2);

                // assign slope
                slopes[i] = (double) kthnumBuckets[i] / (double) (pivotsRight[i] - pivotsLeft[i]);

                if (isinf(slopes[i]))
                    slopes[i] = 0;
            }
        }

        // Assign bucket number and slope to the last active buckets
        if (index < 1) {
            // exclusive cumulative sum to the kthnumbuckets for finding the correct number of buckets
            // for the last active buckets
            thrust::exclusive_scan(thrust::device, kthnumBuckets, kthnumBuckets + numUniqueBuckets, kthnumBuckets, 0);


            // assign slope
            slopes[numUniqueBuckets - 1] = (numBuckets - kthnumBuckets[numUniqueBuckets - 1])
                                           / (double) (pivotsRight[numUniqueBuckets - 1] -
                                                        pivotsLeft[numUniqueBuckets - 1]);

            if (isinf(slopes[numUniqueBuckets - 1]))
                slopes[numUniqueBuckets - 1] = 0;
        }

        __syncthreads();

        // if we have extreme cases
        if (kthnumBuckets[numUniqueBuckets - 1] >= numBuckets) {
            if (index < numUniqueBuckets - 1) {
                for (int i = index; i < numUniqueBuckets - 1; i += offset) {

                    // assign bucket number
                    kthnumBuckets[i] = max(uniqueBucketsCounts[i] * numBuckets / length, 1);

                    // assign slope
                    slopes[i] = (double) kthnumBuckets[i] / (double) (pivotsRight[i] - pivotsLeft[i]);

                    if (isinf(slopes[i]))
                        slopes[i] = 0;
                }
            }

            // Assign bucket number and slope to the last active buckets
            if (index < 1) {
                // exclusive cumulative sum to the kthnumbuckets for finding the correct number of buckets
                // for the last active buckets
                thrust::exclusive_scan(thrust::device, kthnumBuckets, kthnumBuckets + numUniqueBuckets, kthnumBuckets, 0);


                // assign slope
                slopes[numUniqueBuckets - 1] = (numBuckets - kthnumBuckets[numUniqueBuckets - 1])
                                               / (double) (pivotsRight[numUniqueBuckets - 1] -
                                                            pivotsLeft[numUniqueBuckets - 1]);

                if (isinf(slopes[numUniqueBuckets - 1]))
                    slopes[numUniqueBuckets - 1] = 0;
            }
        }
    }



    /* This function assigns elements to buckets based on the pivots and slopes determined
       by a randomized sampling of the elements in the vector. At the same time, this
       function keeps track of count.

       d_elementToBucket : bucket assignment for every array element
       d_bucketCount : number of element that falls into the indexed buckets within the block
    */
    template <typename T>
    __global__ void assignSmartBucket_iterative
            (T * d_vector, int length, unsigned int * d_elementToBucket,
             double * slopes, T * pivotsLeft, T * pivotsRight,
             unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
             int numUniqueBuckets, int numBuckets, int offset) {

        int index = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int bucketIndex;
        int threadIndex = threadIdx.x;

        //variables in shared memory for fast access
        extern __shared__ unsigned int array[];
        double * sharedSlopes = (double *)array;
        T * sharedPivotsLeft = (T *)&sharedSlopes[numUniqueBuckets];
        unsigned int * sharedkthNumBuckets = (unsigned int *)&sharedPivotsLeft[numUniqueBuckets];
        unsigned int * sharedBuckets = (unsigned int *)&sharedkthNumBuckets[numUniqueBuckets];

        //reading bucket counts into shared memory where increments will be performed
        for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) {

            if (threadIndex < numBuckets)
                sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex] = 0;
        }

        if (threadIndex < numUniqueBuckets) {
            sharedPivotsLeft[threadIndex] = pivotsLeft[threadIndex];
            sharedSlopes[threadIndex] = slopes[threadIndex];
            sharedkthNumBuckets[threadIndex] = kthNumBuckets[threadIndex];
        }

        __syncthreads();

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
            }
        }

        __syncthreads();

        //reading bucket counts from shared memory back to global memory
        for (int i = 0; i <(numBuckets / MAX_THREADS_PER_BLOCK); i++)
            if (threadIndex < numBuckets)
                *(d_bucketCount + blockIdx.x * numBuckets
                  + i * MAX_THREADS_PER_BLOCK + threadIndex) =
                        *(sharedBuckets + i * MAX_THREADS_PER_BLOCK + threadIndex);

    } // closes the kernel



    /* This function cumulatively sums the count of every block for a given bucket s.t. the
       last block index holds the total number of elements falling into that bucket all over the
       array.
       updates d_bucketCount
    */
    __global__ void sumCounts(unsigned int * d_bucketCount, const int numBuckets
            , const int numBlocks) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        for(int j=1; j<numBlocks; j++)
            d_bucketCount[index + numBuckets*j] += d_bucketCount[index + numBuckets*(j-1)];
    }


    /* This function reindexes the buckets counts for every block according to the
       accumulated d_reindexCounter counter for the reduced vector.
       updates d_bucketCount
    */
    __global__ void reindexCounts(unsigned int * d_bucketCount, int numBuckets, int numBlocks,
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
    __global__ void copyElements_iterative
            (T * d_vector, T * d_newvector, int lengthOld, unsigned int * elementToBuckets,
             unsigned int * uniqueBuckets, int numUniqueBuckets,
             unsigned int * d_bucketCount, int numBuckets, unsigned int offset) {

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


    /* This function copies the elements of buckets that contain kVals into a newly allocated
       reduced vector space.
       newArray - reduced size vector containing the essential elements
    */

    template <typename T>
    void updatePivots_iterative_CPU
            (T * pivotsLeft, T * newPivotsLeft, T * newPivotsRight,
             double * slopes, unsigned int * kthnumBuckets, unsigned int * uniqueBuckets,
             int numUniqueBuckets, int numUniqueBucketsOld, int offset) {

        // int index = blockIdx.x * blockDim.x + threadIdx.x;
        int index;

        for (index = 0; index < numUniqueBuckets; index++) {
            for (int i = index; i < numUniqueBuckets; i += offset) {
                unsigned int bucket = uniqueBuckets[i];
                int minBucketIndex = 0;
                int maxBucketIndex = numUniqueBucketsOld;
                int midBucketIndex;


                // perform binary search to find kthNumBucket that is greatest s.t. lower than or equal to the bucket
                for (int j = 1; j < numUniqueBucketsOld; j *= 2) {
                    midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;
                    if (bucket >= kthnumBuckets[midBucketIndex])
                        minBucketIndex = midBucketIndex;
                    else
                        maxBucketIndex = midBucketIndex;
                }


                long double invslope=0.0;
                if (slopes[minBucketIndex] == (double)0) {
                    newPivotsLeft[i] = pivotsLeft[minBucketIndex];
                    newPivotsRight[i] = pivotsLeft[minBucketIndex];
                }
                else {
                    invslope = 1/((long double) slopes[minBucketIndex]);
                    newPivotsLeft[i] = (T)((long double) pivotsLeft[minBucketIndex] +
                                           (((long double) (bucket - kthnumBuckets[minBucketIndex])) * invslope));
                    newPivotsRight[i] = (T) ((long double)pivotsLeft[minBucketIndex] +
                                             (((long double) (bucket - kthnumBuckets[minBucketIndex] + 1) * invslope)));
//                                               slopes[bucketIndex]));
                }
            }
        }

        // needs to swap pointers of pivotsLeft with newPivotsLeft, pivotsRight with newPivotsRight
    }


    /* This function copies the elements of buckets that contain kVals into a newly allocated
       reduced vector space.
       newArray - reduced size vector containing the essential elements
    */
    template <typename T>
    __global__ void updatePivots_iterative
            (T * d_pivotsLeft, T * d_newPivotsLeft, T * d_newPivotsRight,
             double * slopes, unsigned int * kthnumBuckets, unsigned int * uniqueBuckets,
             int numUniqueBuckets, int numUniqueBucketsOld, int offset) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < numUniqueBuckets) {
            for (int i = index; i < numUniqueBuckets; i += offset) {
                unsigned int bucket = uniqueBuckets[i];
                int minBucketIndex = 0;
                int maxBucketIndex = numUniqueBucketsOld;
                int midBucketIndex;


                // perform binary search to find kthNumBucket that is greatest s.t. lower than or equal to the bucket
                for (int j = 1; j < numUniqueBucketsOld; j *= 2) {
                    midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;
                    if (bucket >= kthnumBuckets[midBucketIndex])
                        minBucketIndex = midBucketIndex;
                    else
                        maxBucketIndex = midBucketIndex;
                }


                long double invslope=0.0;
                if (slopes[minBucketIndex] == (double)0) {
                    d_newPivotsLeft[i] = d_pivotsLeft[minBucketIndex];
                    d_newPivotsRight[i] = d_pivotsLeft[minBucketIndex];
                }
                else {
                    invslope = 1/((long double) slopes[minBucketIndex]);
                    d_newPivotsLeft[i] = (T)((long double) d_pivotsLeft[minBucketIndex] +
                                             (((long double) (bucket - kthnumBuckets[minBucketIndex])) * invslope)); // / slopes[bucketIndex]));
                    d_newPivotsRight[i] = (T) ((long double)d_pivotsLeft[minBucketIndex] +
                                               (((long double) (bucket - kthnumBuckets[minBucketIndex] + 1) * invslope)));
//                                               slopes[bucketIndex]));
                }
            }
        }

        // needs to swap pointers of pivotsLeft with newPivotsLeft, pivotsRight with newPivotsRight
    }

    /*
     * This function finds the actual element for the kth orderstats by giving the list of buckets
     */
    template <typename T>
    __global__ void updateOutput_iterative
            (T * d_vector, unsigned int * d_elementToBucket, int lengthOld, T * d_tempOutput,
             unsigned int * d_tempKorderBucket, int tempKorderLength, int offset) {

        int index = blockDim.x * blockIdx.x + threadIdx.x;

        if (index < lengthOld) {
            for (int i = index; i < lengthOld; i += offset) {
                unsigned int bucket = d_elementToBucket[i];

                int minBucketIndex = 0;
                int maxBucketIndex = tempKorderLength - 1;
                int midBucketIndex;

                while (minBucketIndex <= maxBucketIndex) {
                    midBucketIndex = minBucketIndex + ((maxBucketIndex - minBucketIndex) / 2);

                    if (d_tempKorderBucket[midBucketIndex] == bucket) {
                        d_tempOutput[midBucketIndex] = d_vector[i];
                        break;
                    }
                    else if (bucket < d_tempKorderBucket[midBucketIndex]) {
                        maxBucketIndex = midBucketIndex - 1;
                    }
                    else if (bucket > d_tempKorderBucket[midBucketIndex]) {
                        minBucketIndex = midBucketIndex + 1;
                    }

                }

            }
        }
    }


    /*
     * This function finds the actual element for the kth orderstats by giving the list of buckets
     */
    template <typename T>
    __global__ void updateOutput_iterative_last
            (T * d_vector, unsigned int * d_elementToBucket, int lengthOld, T * d_tempOutput,
             unsigned int * d_tempKorderBucket, int tempKorderLength, int offset){

        int index = blockDim.x * blockIdx.x + threadIdx.x;

        if (index < lengthOld) {
            for (int i = index; i < lengthOld; i += offset) {
                unsigned int bucket = d_elementToBucket[i];

                for (int j = 0; j < tempKorderLength; j++) {
                    if (d_tempKorderBucket[j] == bucket)
                        d_tempOutput[j] = d_vector[i];
                }
            }
        }
    }




    /// ***********************************************************
    /// ***********************************************************
    /// **** GENERATE KD PIVOTS
    /// ***********************************************************
    /// ***********************************************************

    /* Hash function using Monte Carlo method
     */
    __host__ __device__
    unsigned int hash(unsigned int a) {
        a = (a+0x7ed55d16) + (a<<12);
        a = (a^0xc761c23c) ^ (a>>19);
        a = (a+0x165667b1) + (a<<5);
        a = (a+0xd3a2646c) ^ (a<<9);
        a = (a+0xfd7046c5) + (a<<3);
        a = (a^0xb55a4f09) ^ (a>>16);
        return a;
    }



    /* This function creates a random vector of 1024 elements in the range [0 1]
     */
    template <typename T>
    void createRandomVector(T * d_vec, int size) {
        timeval t1;
        unsigned int seed;

        gettimeofday(&t1, NULL);
        seed = t1.tv_usec * t1.tv_sec;
        // seed = 1000000000;

        thrust::device_ptr<T> d_ptr(d_vec);
        thrust::transform (thrust::counting_iterator<unsigned int>(0),
                           thrust::counting_iterator<unsigned int>(size),
                           d_ptr, RandomNumberFunctor(seed));
    }



    /* This function maps the [0 1] range to the [0 vectorSize] and
       grabs the corresponding elements.
    */
    template <typename T>
    __global__ void enlargeIndexAndGetElements (T * in, T * list, int size) {
        *(in + blockIdx.x*blockDim.x + threadIdx.x) =
                *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
    }

    __global__ void enlargeIndexAndGetElements (float * in, int * out, int * list, int size) {
        *(out + blockIdx.x * blockDim.x + threadIdx.x) =
                (int) *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
    }


    __global__ void enlargeIndexAndGetElements
            (float * in, unsigned int * out, unsigned int * list, int size) {
        *(out + blockIdx.x * blockDim.x + threadIdx.x) =
                (unsigned int) *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
    }



    /* This function generates Pivots from the random sampled data and calculates slopes.

       pivots - arrays of pivots
       slopes - array of slopes
    */
    template <typename T>
    void generatePivots (int * pivots, double * slopes, int * d_list, int sizeOfVector
            , int numPivots, int sizeOfSample, int totalSmallBuckets, int min, int max) {

        float * d_randomFloats;
        int * d_randomInts;
        int endOffset = 22;
        int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
        int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

        cudaMalloc (&d_randomFloats, sizeof (float) * sizeOfSample);

        d_randomInts = (int *) d_randomFloats;

        createRandomVector (d_randomFloats, sizeOfSample);

        // converts randoms floats into elements from necessary indices
        enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
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





    /* This function generates Pivots from the random sampled data and calculates slopes.

       pivots - arrays of pivots
       slopes - array of slopes
    */
    template <typename T>
    void generatePivots (unsigned int * pivots, double * slopes, unsigned int * d_list, int sizeOfVector
            , int numPivots, int sizeOfSample, int totalSmallBuckets, unsigned int min, unsigned int max) {

        float * d_randomFloats;
        unsigned int * d_randomInts;
        int endOffset = 22;
        int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
        int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

        cudaMalloc (&d_randomFloats, sizeof (float) * sizeOfSample);

        d_randomInts = (unsigned int *) d_randomFloats;

        createRandomVector (d_randomFloats, sizeOfSample);

        // converts randoms floats into elements from necessary indices
        enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
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

    template <typename T>
    void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector
            , int numPivots, int sizeOfSample, int totalSmallBuckets, T min, T max) {
        T * d_randoms;
        int endOffset = 22;
        int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
        int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

        cudaMalloc (&d_randoms, sizeof (T) * sizeOfSample);

        createRandomVector (d_randoms, sizeOfSample);

        // converts randoms floats into elements from necessary indices
        enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
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




    /// ***********************************************************
    /// ***********************************************************
    /// **** iterativeSMOS: the main algorithm
    /// ***********************************************************
    /// ***********************************************************


    /* This function is the main process of the algorithm. It reduces the given multi-selection
       problem to a smaller problem by using bucketing ideas.
    */
    template <typename T>
    T iterativeSMOS (T* d_vector, int length, unsigned int * kVals, int numKs, T * output, int blocks
            , int threads, int numBuckets, int numPivots) {

        /// ***********************************************************
        /// **** STEP 1: Initialization
        /// **** STEP 1.1: Find Min and Max of the whole vector
        /// **** We don't need to go through the rest of the algorithm if it's flat
        /// ***********************************************************

        //find max and min with thrust
        T maximum, minimum;

        thrust::device_ptr<T>dev_ptr(d_vector);
        thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result =
                thrust::minmax_element(dev_ptr, dev_ptr + length);

        cudaThreadSynchronize();

        minimum = *result.first;
        maximum = *result.second;

        //if the max and the min are the same, then we are done
        if (maximum == minimum) {
            for (int i = 0; i < numKs; i++)
                output[i] = minimum;

            return 1;
        }

        SAFEcuda("End of STEP 1.1\n");


        /// ***********************************************************
        /// **** STEP 1: Initialization
        /// **** STEP 1.2: Declare variables and allocate memory
        /// **** Declare Variables
        /// ***********************************************************

        // declare variables for kernel launches
        int threadsPerBlock = threads;
        int numBlocks = blocks;
        int offset = blocks * threads;

        // variables for the randomized selection
        int sampleSize = 1024;

        // pivots variables
        // potential to simplify
        int numMemory;
        if (numKs > numPivots)
            numMemory = numKs;
        else
            numMemory = numPivots;  // replace this with max

        double * slopes = (double*)malloc(numMemory * sizeof(double));                  // size will be different
        double * d_slopes;
        T * pivots = (T*)malloc(numPivots * sizeof(T));
        T * d_pivots;
        CUDA_CALL(cudaMalloc(&d_slopes, numMemory * sizeof(double)));
        CUDA_CALL(cudaMalloc(&d_pivots, numPivots * sizeof(T)));

        T * pivotsLeft = (T*)malloc(numMemory * sizeof(T));                                 // new variables
        T * pivotsRight = (T*)malloc(numMemory * sizeof(T));
        // T * newPivotsLeft = (T*)malloc(numMemory * sizeof(T));								// test part
        // T * newPivotsRight = (T*)malloc(numMemory * sizeof(T));								// test part
        T * d_pivotsLeft;
        T * d_pivotsRight;
        T * d_newPivotsLeft;
        T * d_newPivotsRight;
        CUDA_CALL(cudaMalloc(&d_pivotsLeft, numMemory * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_pivotsRight, numMemory * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_newPivotsLeft, numMemory * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_newPivotsRight, numMemory * sizeof(T)));


        //Allocate memory to store bucket assignments
        size_t size = length * sizeof(unsigned int);
        unsigned int * d_elementToBucket;    //array showing what bucket every element is in
        CUDA_CALL(cudaMalloc(&d_elementToBucket, size));


        // Allocate memory to store bucket counts
        size_t totalBucketSize = numBlocks * numBuckets * sizeof(unsigned int);
        unsigned int * h_bucketCount = (unsigned int *) malloc (numBuckets * sizeof (unsigned int));
        //array showing the number of elements in each bucket
        unsigned int * d_bucketCount;
        CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));


        // Allocate memory to store the new vector for kVals
        T * d_newvector;
        CUDA_CALL(cudaMalloc(&d_newvector, length * sizeof(T)));
        T * addressOfd_newvector = d_newvector;


        // array of kth buckets
        int numUniqueBuckets;
        int numUniqueBucketsOld;
        int lengthOld;
        int tempKorderLength;
        unsigned int * d_kVals;
        unsigned int * kthBuckets = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_kthBuckets;
        unsigned int * kthBucketScanner = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * kIndices = (unsigned int *)malloc(numKs * sizeof(unsigned int));
        unsigned int * d_kIndices;
        unsigned int * uniqueBuckets = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_uniqueBuckets;
        unsigned int * uniqueBucketCounts = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_uniqueBucketCounts;
        unsigned int * reindexCounter = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_reindexCounter;
        unsigned int * kthnumBuckets = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_kthnumBuckets;
        T * tempOutput = (T *)malloc(numMemory * sizeof(T));
        T * d_tempOutput;
        unsigned int * tempKorderBucket = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_tempKorderBucket;
        unsigned int * tempKorderIndeces = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_tempKorderIndeces;
        CUDA_CALL(cudaMalloc(&d_kVals, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_kIndices, numKs * sizeof (unsigned int)));
        CUDA_CALL(cudaMalloc(&d_kthBuckets, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_uniqueBuckets, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_uniqueBucketCounts, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_reindexCounter, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_kthnumBuckets, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_tempOutput, numMemory * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_tempKorderBucket, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_tempKorderIndeces, numMemory * sizeof(unsigned int)));

        for (int i = 0; i < numMemory; i++) {
            kthBucketScanner[i] = 0;
            // kIndices[i] = i;
        }

        for (int i = 0; i < numKs; i++) {
            kIndices[i] = i;
        }
        
        // decide if algorithm need to ignore updating order statistics in first iteration
        bool doNotUpdate = false;
        if (length >= CUTOFF)
        	doNotUpdate = true;
        	

        SAFEcuda("End of STEP 1.2\n");


        /// ***********************************************************
        /// **** STEP 1: Initialization
        /// **** STEP 1.3: Sort the klist
        /// **** and we have to keep the old index
        /// ***********************************************************

        CUDA_CALL(cudaMemcpy(d_kIndices, kIndices, numKs * sizeof (unsigned int),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_kVals, kVals, numKs * sizeof (unsigned int),
                             cudaMemcpyHostToDevice));

        // sort the given indices
        thrust::device_ptr<unsigned int>kVals_ptr(d_kVals);
        thrust::device_ptr<unsigned int>kIndices_ptr(d_kIndices);
        thrust::sort_by_key(kVals_ptr, kVals_ptr + numKs, kIndices_ptr);

        cudaThreadSynchronize();

        CUDA_CALL(cudaMemcpy(kIndices, d_kIndices, numKs * sizeof (unsigned int),
                             cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(kVals, d_kVals, numKs * sizeof (unsigned int),
                             cudaMemcpyDeviceToHost));

        SAFEcuda("End of STEP 1.3\n");

        cudaThreadSynchronize();

        /// ***********************************************************
        /// **** STEP 2: CreateBuckets
        /// ****  Declare and Generate Pivots and Slopes
        /// ***********************************************************

        // Find bucket sizes using a randomized selection
        generatePivots<T>(pivots, slopes, d_vector, length, numPivots, sampleSize,
                          numBuckets, minimum, maximum);

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
        }
        numUniqueBuckets = numPivots - 1;

        cudaThreadSynchronize();


        CUDA_CALL(cudaMemcpy(d_slopes, slopes, (numPivots - 1) * sizeof(double),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_pivotsLeft, pivotsLeft, numUniqueBuckets * sizeof(T),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_pivotsRight, pivotsRight, numUniqueBuckets * sizeof(T),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_kthnumBuckets, kthnumBuckets, numUniqueBuckets * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));

        SAFEcuda("End of STEP 2\n");



        /// ***********************************************************
        /// **** STEP 3: AssignBuckets
        /// **** Using the function assignSmartBucket
        /// ***********************************************************


        assignSmartBucket_iterative<T><<<numBlocks, threadsPerBlock, numUniqueBuckets * sizeof(T) +
                                                                     numUniqueBuckets * sizeof(double) +
                                                                     numUniqueBuckets * sizeof(unsigned int) +
                                                                     numBuckets * sizeof(unsigned int)>>>
        (d_vector, length, d_elementToBucket, d_slopes, d_pivotsLeft, d_pivotsRight,
                d_kthnumBuckets, d_bucketCount, numUniqueBuckets, numBuckets, offset);


        cudaThreadSynchronize();



        SAFEcuda("End of STEP 3\n");


        /// ***********************************************************
        /// **** STEP 4: IdentifyActiveBuckets
        /// **** Find the kth buckets
        /// **** and update their respective indices
        /// ***********************************************************

        // count number of elements in each buckets
        sumCounts<<<numBuckets/threadsPerBlock, threadsPerBlock>>>(d_bucketCount, numBuckets, numBlocks);

        cudaThreadSynchronize();

        SAFEcuda("STEP 4, after sumCounts\n");

        // update order statistics and buckets
        findKBuckets(d_bucketCount, h_bucketCount, numBuckets, kVals, numKs, kthBucketScanner, kthBuckets, numBlocks);

        SAFEcuda("STEP 4, after findKBuckets");

        // update order statistics and buckets
        updatekVals<T>
                (kVals, &numKs, output, kIndices, &length, &lengthOld, h_bucketCount, kthBuckets, kthBucketScanner,
                 reindexCounter, uniqueBuckets, uniqueBucketCounts, &numUniqueBuckets, &numUniqueBucketsOld);

        SAFEcuda("STEP 4, after updatekVals\n");

        cudaThreadSynchronize();

        bool whetherEnterLoop = true;
        if (numKs <= 0)
            whetherEnterLoop = false;

        int numLengthEqual = 0;

        /// ***********************************************************
        /// **** STEP 5: Reduce
        /// **** Iteratively go through the loop to find correct
        /// **** order statistics and reduce the vector size
        /// ***********************************************************

        for (int j = 0; j < 50 && whetherEnterLoop; j++) {

            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.1: Copy active elements
            /// **** Copy the elements from the unique active buckets
            /// ***********************************************************

            CUDA_CALL(cudaMemcpy(d_reindexCounter, reindexCounter,
                                 numUniqueBuckets * sizeof(unsigned int), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_uniqueBuckets, uniqueBuckets,
                                 numUniqueBuckets * sizeof(unsigned int), cudaMemcpyHostToDevice));

            // perform reindex counter function
            reindexCounts<<<(int) ceil((float)numUniqueBuckets/threadsPerBlock), threadsPerBlock>>>
                    (d_bucketCount, numBuckets, numBlocks, d_reindexCounter, d_uniqueBuckets, numUniqueBuckets);

            SAFEcuda("STEP 5.1, after reindexCounts\n");

            cudaThreadSynchronize();

            // copy elements from active buckets
            copyElements_iterative<T><<<numBlocks, threadsPerBlock, numUniqueBuckets * sizeof(unsigned int)>>>
            (d_vector, d_newvector, lengthOld, d_elementToBucket, d_uniqueBuckets, numUniqueBuckets,
                    d_bucketCount, numBuckets, offset);

            SAFEcuda("STEP 5.1, after copyElements\n");

            // swap pointers
            swapPointers(&d_vector, &d_newvector);

            SAFEcuda("STEP 5.1, after swapPointers\n");

            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.2: Update the pivots
            /// **** Update pivots to generate Pivots and Slopes in Step 5.3
            /// ***********************************************************

            CUDA_CALL(cudaMemcpy(d_uniqueBuckets, uniqueBuckets, numUniqueBuckets * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice));

            // potential to fix how many blocks to assign
            updatePivots_iterative<T><<<(int) ceil((float)numUniqueBuckets/threadsPerBlock), threadsPerBlock>>>(d_pivotsLeft, d_newPivotsLeft, d_newPivotsRight,
                    d_slopes, d_kthnumBuckets, d_uniqueBuckets,
                    numUniqueBuckets, numUniqueBucketsOld, offset);

            cudaThreadSynchronize();

            SAFEcuda("STEP 5.2, after updatePivots\n");

            swapPointers(&d_pivotsLeft, &d_newPivotsLeft);
            swapPointers(&d_pivotsRight, &d_newPivotsRight);

            SAFEcuda("STEP 5.2, after swapPointers\n");

            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.3: create slopes and buckets offset
            /// **** create slopes and buckets offset for next iteration
            /// ***********************************************************

            CUDA_CALL(cudaMemcpy(d_uniqueBucketCounts, uniqueBucketCounts, numUniqueBuckets * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice));

            // potential to fix how many blocks to assign
            generateBucketsandSlopes_iterative<<<(int) ceil((float)numUniqueBuckets/threadsPerBlock), threadsPerBlock>>>
                    (d_pivotsLeft, d_pivotsRight, d_slopes, d_uniqueBucketCounts,
                     numUniqueBuckets, d_kthnumBuckets, length, offset, numBuckets);

            cudaThreadSynchronize();

            SAFEcuda("STEP 5.3, after generateBucketandSlopes\n");

            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.4: assign buckets
            /// **** assign elements to correct buckets in iteration
            /// ***********************************************************

            // assign index of buckets to each number
            assignSmartBucket_iterative<T><<<numBlocks, threadsPerBlock, numUniqueBuckets * sizeof(T) +
                                                                         numUniqueBuckets * sizeof(double) +
                                                                         numUniqueBuckets * sizeof(unsigned int) +
                                                                         numBuckets * sizeof(unsigned int)>>>
            (d_vector, length, d_elementToBucket, d_slopes, d_pivotsLeft, d_pivotsRight, d_kthnumBuckets,
                    d_bucketCount, numUniqueBuckets, numBuckets, offset);

            cudaThreadSynchronize();

            SAFEcuda("STEP 5.4, after assignSmartBucket\n");

            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.5: IdentifyActiveBuckets
            /// **** Find kth buckets and update their respective indices
            /// ***********************************************************

            cudaDeviceSynchronize();

            sumCounts<<<numBuckets/threadsPerBlock, threadsPerBlock>>>(d_bucketCount, numBuckets, numBlocks);

            SAFEcuda("STEP 5.5, after sumCounts\n");

            cudaThreadSynchronize();

            findKBuckets(d_bucketCount, h_bucketCount, numBuckets, kVals, numKs, kthBucketScanner, kthBuckets, numBlocks);

            SAFEcuda("STEP 5.5, after findKBuckets\n");

            // if vector is beyond the CUTOFF point, we will not update order in first iteration
			if (j == 0 && doNotUpdate) {
				updatekVals<T>
		            (kVals, &numKs, output, kIndices, &length, &lengthOld, 
		             h_bucketCount, kthBuckets, kthBucketScanner, reindexCounter, 
		             uniqueBuckets, uniqueBucketCounts, &numUniqueBuckets, 
		             &numUniqueBucketsOld);
			
			}
			else {
		        updatekVals_iterative<T>(kVals, &numKs, output, kIndices, &length, &lengthOld, h_bucketCount, 
		        						 kthBuckets, kthBucketScanner, reindexCounter, uniqueBuckets, uniqueBucketCounts, 
		        						 &numUniqueBuckets, &numUniqueBucketsOld, tempKorderBucket, tempKorderIndeces, 
		        						 &tempKorderLength);

		        SAFEcuda("STEP 5.5, after updateKVals\n");

                // if we found any order statistics
		        if (tempKorderLength > 0) {
		            cudaThreadSynchronize();

		            CUDA_CALL(cudaMemcpy(d_tempKorderBucket, tempKorderBucket, tempKorderLength * sizeof(unsigned int),
		                                 cudaMemcpyHostToDevice));
		            CUDA_CALL(cudaMemcpy(d_tempKorderIndeces, tempKorderIndeces, tempKorderLength * sizeof(unsigned int),
		                                 cudaMemcpyHostToDevice));

		            // potential to fix how many blocks to assign
		            updateOutput_iterative_last<<<(int) ceil((float)lengthOld/threadsPerBlock), threadsPerBlock>>>
		                    (d_vector, d_elementToBucket, lengthOld, d_tempOutput, d_tempKorderBucket, tempKorderLength, offset);

		            SAFEcuda("STEP 5.5, after updateOutput\n");

		            CUDA_CALL(cudaMemcpy(tempOutput, d_tempOutput, tempKorderLength * sizeof(T),
		                                 cudaMemcpyDeviceToHost));

		            for (int i = 0; i < tempKorderLength; i++)
		                output[tempKorderIndeces[i]] = tempOutput[i];

		        }
		    }



            if (length == lengthOld) {
                numLengthEqual++;
                if (numLengthEqual > 2 || length == 0 || numKs == 0)
                    break;
            }
            else {
                numLengthEqual = 0;
            }


            cudaThreadSynchronize();
        }

        // update all repeated order statistics in the end
        if (numKs > 0) {
            cudaThreadSynchronize();

            CUDA_CALL(cudaMemcpy(d_kthBuckets, kthBuckets, numKs * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice));

            updateOutput_iterative_last<<<(int) ceil((float)lengthOld/threadsPerBlock), threadsPerBlock>>>
                    (d_vector, d_elementToBucket, lengthOld, d_tempOutput, d_kthBuckets, numKs, offset);

            SAFEcuda("Exit Iteration, after updateOutput\n");

            CUDA_CALL(cudaMemcpy(tempOutput, d_tempOutput, numKs * sizeof(T),
                                 cudaMemcpyDeviceToHost));

            for (int i = 0; i < numKs; i++)
                output[kIndices[i]] = tempOutput[i];
        }

        cudaThreadSynchronize();

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

        SAFEcuda("Exit Iteration, after free\n");

        cudaThreadSynchronize();


        return 0;
    }

    template <typename T>
    T iterativeSMOSWrapper (T * d_vector, int length, uint * kVals_ori, int numKs
            , T * outputs, int blocks, int threads) {

        int numBuckets = 8192;
        unsigned int * kVals = (unsigned int *)malloc(numKs * sizeof(unsigned int));

        // turn it into kth smallest
        for (register int i = 0; i < numKs; i++)
            kVals[i] = length - kVals_ori[i] + 1;

        iterativeSMOS(d_vector, length, kVals, numKs, outputs, blocks, threads, numBuckets, 17);

        cudaThreadSynchronize();

        // test part
        // printf("k[414]: %.10f\n", outputs[414]);

        free(kVals);

        cudaThreadSynchronize();

        return 1;
    }

    template int iterativeSMOSWrapper (int * d_vector, int length, uint * kVals_ori, int numKs,
                                       int * outputs, int blocks, int threads);
    template unsigned int iterativeSMOSWrapper
            (unsigned int * d_vector, int length, uint * kVals_ori, int numKs,
             unsigned int * outputs, int blocks, int threads);
    template float iterativeSMOSWrapper (float * d_vector, int length, uint * kVals_ori, int numKs,
                                         float * outputs, int blocks, int threads);
    template double iterativeSMOSWrapper (double * d_vector, int length, uint * kVals_ori, int numKs,
                                          double * outputs, int blocks, int threads);
}

int cmpfunc (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}
