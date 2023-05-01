#ifndef ITERATIVESMOS_HPP
#define ITERATIVESMOS_HPP

namespace IterativeSMOS {

	/// ***********************************************************
    /// ***********************************************************
    /// **** HELPER SAFE FUNCTIONS
    /// ***********************************************************
    /// ***********************************************************
	void check_malloc_int(int *pointer, const char *message);
	
	void check_malloc_float(float *pointer, const char *message);
	
	void check_malloc_double(double *pointer, const char *message);
	
	void check_cudaMalloc(const char *message);
	
	void Check_CUDA_Error(const char *message);

	void check_cudaMalloc2(cudaError_t status, const char *message);
	
	inline void SAFEcudaMalloc2(cudaError_t status, const char *message);

	inline void SAFEcudaMalloc(const char *message);
	
	inline void SAFEcuda(const char *message);

	inline void SAFEmalloc_int(int * pointer, const char *message);
	
	inline void SAFEmalloc_float(float *pointer, const char *message);
	
	inline void SAFEmalloc_double(double *pointer, const char *message);
	
	template<typename T>
    void setToAllZero (T * d_vector, int length);
    
    
    
    /// ***********************************************************
    /// ***********************************************************
    /// **** HELPER CPU FUNCTIONS
    /// ***********************************************************
    /// ***********************************************************
    
    template<typename T>
    void setToAllZero (T * d_vector, int length);
    
    inline int findKBuckets(unsigned int * d_bucketCount, unsigned int * h_bucketCount, int numBuckets
            , const unsigned int * kVals, int numKs, unsigned int * sums, unsigned int * markedBuckets
            , int numBlocks);
            
    template <typename T>
    inline int updatekVals_iterative
    		(unsigned int * kVals, int * numKs, T * output, unsigned int * kIndicies,
             int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
             unsigned int * kthBucketScanner, unsigned int * reindexCounter,
             unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
             int * numUniqueBuckets, int * numUniqueBucketsOld,
             unsigned int * tempKorderBucket, unsigned int * tempKorderIndeces, int * tempKorderLength);
             
             
    /// ***********************************************************
    /// ***********************************************************
    /// **** HELPER GPU FUNCTIONS-KERNELS
    /// ***********************************************************
    /// ***********************************************************
    
    template <typename T>
    __global__ void generateBucketsandSlopes_iterative 
    		(T * pivotsLeft, T * pivotsRight, double * slopes,
             unsigned int * uniqueBucketsCounts, int numUniqueBuckets,
             unsigned int * kthnumBuckets, int length, int offset, int numBuckets);
    
    /*
    __global__ void generateBucketsandSlopes_iterative 
    		(double * pivotsLeft, double * pivotsRight, double * slopes,
             unsigned int * uniqueBucketsCounts, int numUniqueBuckets,
             unsigned int * kthnumBuckets, int length, int offset, int numBuckets);
    */
    
     
     template <typename T>
    __global__ void assignSmartBucket_iterative
    		(T * d_vector, int length, unsigned int * d_elementToBucket,
             double * slopes, T * pivotsLeft, T * pivotsRight,
             unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
             int numUniqueBuckets, int numBuckets, int offset);
     
     
    /*
    template <typename T>
    __global__ void assignSmartBucket_iterative
    		(T * d_vector, int length, unsigned int * d_elementToBucket,
             double * slopes, double * pivotsLeft, double * pivotsRight,
             unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
             int numUniqueBuckets, int numBuckets, int offset);
     */
                
     __global__ void sumCounts(unsigned int * d_bucketCount, const int numBuckets
            , const int numBlocks);
            
     __global__ void reindexCounts(unsigned int * d_bucketCount, int numBuckets, int numBlocks,
                                  unsigned int * d_reindexCounter, unsigned int * d_uniqueBuckets,
                                  const int numUniqueBuckets);
                                  
     
     template <typename T>
    __global__ void copyElements_iterative 
    		(T * d_vector, T * d_newvector, int lengthOld, unsigned int * elementToBuckets,
             unsigned int * uniqueBuckets, int numUniqueBuckets,
             unsigned int * d_bucketCount, int numBuckets, unsigned int offset);
            
    /*
    template <typename T>
    __global__ void updatePivots_iterative
    		(T * d_pivotsLeft, T * d_newPivotsLeft, T * d_newPivotsRight,
             double * slopes, unsigned int * kthnumBuckets, unsigned int * uniqueBuckets,
             int numUniqueBuckets, int numUniqueBucketsOld, int offset);
     */
     /*
     __global__ void updatePivots_iterative
		(double * d_pivotsLeft, double * d_newPivotsLeft, double * d_newPivotsRight,
         double * slopes, unsigned int * kthnumBuckets, unsigned int * uniqueBuckets,
         int numUniqueBuckets, int numUniqueBucketsOld, int offset);
     */
    
    /*
    template <typename T>
	void updatePivots_iterative_CPU
    		(T * pivotsLeft, T * newPivotsLeft, T * newPivotsRight,
             double * slopes, unsigned int * kthnumBuckets, unsigned int * uniqueBuckets,
             int numUniqueBuckets, int numUniqueBucketsOld, int offset);
     */
     
	void updatePivots_iterative_CPU
    		(double * pivotsLeft, double * newPivotsLeft, double * newPivotsRight,
             double * slopes, unsigned int * kthnumBuckets, unsigned int * uniqueBuckets,
             int numUniqueBuckets, int numUniqueBucketsOld, int offset);
    
	template <typename T>
    __global__ void updateOutput_iterative 
    		(T * d_vector, unsigned int * d_elementToBucket, int lengthOld, T * d_tempOutput,
             unsigned int * d_tempKorderBucket, int tempKorderLength, int offset);
	
	template <typename T>
    __global__ void updateOutput_iterative_last 
    		(T * d_vector, unsigned int * d_elementToBucket, int lengthOld, T * d_tempOutput,
             unsigned int * d_tempKorderBucket, int tempKorderLength, int offset);
	
    /// ***********************************************************
    /// ***********************************************************
    /// **** GENERATE KD PIVOTS
    /// ***********************************************************
    /// ***********************************************************
    
    __host__ __device__
    unsigned int hash(unsigned int a);
    
    struct RandomNumberFunctor :
            public thrust::unary_function<unsigned int, float> {
        unsigned int mainSeed;

        RandomNumberFunctor(unsigned int _mainSeed) :
                mainSeed(_mainSeed) {}

        __host__ __device__
        float operator()(unsigned int threadIdx)
        {
            unsigned int seed = hash(threadIdx) * mainSeed;

            thrust::default_random_engine rng(seed);
            rng.discard(threadIdx);
            thrust::uniform_real_distribution<float> u(0, 1);

            return u(rng);
        }
    };
    
    
    template <typename T>
    void createRandomVector(T * d_vec, int size);
    
    template <typename T>
    __global__ void enlargeIndexAndGetElements(T * in, T * list, int size);
    
    __global__ void enlargeIndexAndGetElements (float * in, int * out, int * list, int size);
    
    __global__ void enlargeIndexAndGetElements 
    		(float * in, unsigned int * out, unsigned int * list, int size);
    
    
    template <typename T>
    void generatePivots (int * pivots, double * slopes, int * d_list, int sizeOfVector
            , int numPivots, int sizeOfSample, int totalSmallBuckets, int min, int max);
            
    template <typename T>
    void generatePivots 
    	(unsigned int * pivots, double * slopes, unsigned int * d_list, int sizeOfVector,
         int numPivots, int sizeOfSample, int totalSmallBuckets, unsigned int min, unsigned int max);
         
    template <typename T>
    void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector
            , int numPivots, int sizeOfSample, int totalSmallBuckets, T min, T max);
    
    
    
    /// ***********************************************************
    /// ***********************************************************
    /// **** iterativeSMOS: the main algorithm
    /// ***********************************************************
    /// ***********************************************************
    
    template <typename T>
    T iterativeSMOS (T* d_vector, int length, unsigned int * kVals, int numKs, T * output, int blocks
            , int threads, int numBuckets, int numPivots);
            
    template <typename T>
    T iterativeSMOSWrapper (T * d_vector, int length, uint * kVals_ori, int numKs
            , T * outputs, int blocks, int threads);
            
    int cmpfunc (const void * a, const void * b);
    
}


#endif
