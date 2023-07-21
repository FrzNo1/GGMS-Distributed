#ifndef DISTRIBUTEDSMOS_KERNEL_CUH
#define DISTRIBUTEDSMOS_KERNEL_CUH

template <typename T>
void minmax_element_CALL(T* d_vector_local, int length_local, T* maximum_local, T* minimum_local);

void sort_by_key_CALL(unsigned int* d_kVals, unsigned int* d_kIndices, int numKs);

int findKBuckets(unsigned int * h_bucketCount, int numBuckets, 
				const unsigned int * kVals, int numKs, unsigned int * sums, 
				unsigned int * markedBuckets, int numBlocks);
					
template <typename T>
int updatekVals_distributive
			(unsigned int * kVals, int * numKs, T * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld,
             unsigned int * tempKorderBucket, unsigned int * tempKorderIndeces, int * tempKorderLength);

template <typename T>
int updatekVals
			(unsigned int * kVals, int * numKs, T * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld);
             
int updateReindexCounter_distributive
			(unsigned int* reindexCounter, unsigned int* h_bucketCount, 
			 unsigned int* uniqueBuckets, int* length, int* length_Old, 
			 int numUniqueBuckets);
             
template <typename T>
void swapPointers(T** a, T** b);

template <typename T>
T absolute(T a);

template <typename T>
void generateSamples_distributive_CALL
			(T* d_vector, T* d_sampleVector, int length_local, int sampleSize_local, int width);
					 
					 
template <typename T>
void generateBucketsandSlopes_distributive_CALL
			(T * pivotsLeft, T * pivotsRight, double * slopes,
             unsigned int * uniqueBucketsCounts, int numUniqueBuckets,
             unsigned int * kthnumBuckets, int length, int offset, 
             int numBuckets, int threadsPerBlock);
					 
template <typename T>
void assignSmartBucket_distributive_CALL
			(T * d_vector, int length, unsigned int * d_elementToBucket,
             double * slopes, T * pivotsLeft, T * pivotsRight,
             unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
             int numUniqueBuckets, int numBuckets, int offset, 
             int numBlocks, int threadsPerBlock);

void sumCounts_CALL(unsigned int * d_bucketCount, const int numBuckets, 
					const int numBlocks, int threadsPerBlock);
					
void reindexCounts_CALL(unsigned int * d_bucketCount, int numBuckets, int numBlocks,
                        unsigned int * d_reindexCounter, unsigned int * d_uniqueBuckets,
                        const int numUniqueBuckets, int threadsPerBlock);

template <typename T>
void copyElements_distributive_CALL
			(T * d_vector, T * d_newvector, int lengthOld, 
			 unsigned int * elementToBuckets, unsigned int * uniqueBuckets, 
			 int numUniqueBuckets, unsigned int * d_bucketCount, 
			 int numBuckets, unsigned int offset, int threadsPerBlock,
			 int numBlocks);
			 
template <typename T>
void updatePivots_distributive_CALL
			(T * d_pivotsLeft, T * d_newPivotsLeft, T * d_newPivotsRight,
             double * slopes, unsigned int * kthnumBuckets, unsigned int * uniqueBuckets,
             int numUniqueBuckets, int numUniqueBucketsOld, int offset, 
             int threadsPerBlocks);
             
template <typename T>
void updateOutput_distributive_CALL
			(T * d_vector, unsigned int * d_elementToBucket, int lengthOld, 
			 T * d_tempOutput, unsigned int * d_tempKorderBucket, 
			 int tempKorderLength, int offset, int threadsPerBlock);
			 
			 
template <typename T> 
void generatePivots(int * pivots, double * slopes, int * d_list, 
					int sizeOfVector, int numPivots, int sizeOfSample, 
					int totalSmallBuckets, int min, int max);
					
template <typename T>
void generatePivots(unsigned int * pivots, double * slopes, unsigned int * d_list, 
					int sizeOfVector, int numPivots, int sizeOfSample, 
					int totalSmallBuckets, unsigned int min, unsigned int max);
        			
template <typename T>
void generatePivots (T * pivots, double * slopes, T * d_list, 
					 int sizeOfVector, int numPivots, int sizeOfSample, 
					 int totalSmallBuckets, T min, T max);

#endif
