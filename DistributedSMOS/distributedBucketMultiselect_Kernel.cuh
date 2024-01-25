#ifndef DISTRIBUTEDBUCKETMULTISELECT_KERNEL_CUH
#define DISTRIBUTEDBUCKETMULTISELECT_KERNEL_CUH

template <typename T>
void minmax_element_CALL_B(T* d_vector_local, int length_local, T* maximum_local, T* minimum_local);

void sort_by_key_CALL_B(unsigned int* d_kVals, unsigned int* d_kIndices, int numKs);

template <typename T>
void sort_vector_CALL_B(T* d_input, int length);

int findKBuckets_B(unsigned int * h_bucketCount, int numBuckets, 
				const unsigned int * kVals, int numKs, unsigned int * sums, 
				unsigned int * markedBuckets, int numBlocks);
					
template <typename T>
int updatekVals_distributive_B
			(unsigned int * kVals, int * numKs, T * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld);
             
int updateReindexCounter_distributive_B
			(unsigned int* reindexCounter, unsigned int* h_bucketCount, 
			 unsigned int* uniqueBuckets, int* length, int* length_Old, 
			 int numUniqueBuckets);
             
template <typename T>
void swapPointers_B(T** a, T** b);

template <typename T>
void generateSamples_distributive_CALL_B
			(T* d_vector, T* d_sampleVector, int length_local, int sampleSize_local);

template <typename T>
void assignSmartBucket_distributive_CALL_B
			(T * d_vector, int length, unsigned int * d_elementToBucket,
             double * slopes, T * pivotsLeft, T * pivotsRight,
             unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
             int numUniqueBuckets, int numBuckets, int offset, 
             int numBlocks, int threadsPerBlock);

void sumCounts_CALL_B(unsigned int * d_bucketCount, const int numBuckets, 
					const int numBlocks, int threadsPerBlock);
					
void reindexCounts_CALL_B(unsigned int * d_bucketCount, int numBuckets, int numBlocks,
                        unsigned int * d_reindexCounter, unsigned int * d_uniqueBuckets,
                        const int numUniqueBuckets, int threadsPerBlock);

template <typename T>
void copyElements_distributive_CALL_B
			(T * d_vector, T * d_newvector, int lengthOld, 
			 unsigned int * elementToBuckets, unsigned int * uniqueBuckets, 
			 int numUniqueBuckets, unsigned int * d_bucketCount, 
			 int numBuckets, unsigned int offset, int threadsPerBlock,
			 int numBlocks);
			 
template <typename T>
void copyValuesInChunk_CALL_B(T * outputVector, T * inputVector, uint * kList, 
                            uint * kIndices, int kListCount, int numBlocks, 
                            int threadsPerBlock);
					
					
					
					
template <typename T> 
void generatePivots_B(int * pivots, double * slopes, int * d_list, 
					int sizeOfVector, int numPivots, int sizeOfSample, 
					int totalSmallBuckets, int min, int max);
					
template <typename T>
void generatePivots_B(unsigned int * pivots, double * slopes, unsigned int * d_list, 
					int sizeOfVector, int numPivots, int sizeOfSample, 
					int totalSmallBuckets, unsigned int min, unsigned int max);
        			
template <typename T>
void generatePivots_B (T * pivots, double * slopes, T * d_list, 
					 int sizeOfVector, int numPivots, int sizeOfSample, 
					 int totalSmallBuckets, T min, T max);

#endif
