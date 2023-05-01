#ifndef BUCKETMULTISELECT_HPP
#define BUCKETMULTISELECT_HPP

namespace BucketMultiselect {

	/// ***********************************************************
	/// ***********************************************************
	/// **** HELPER CPU FUNCTIONS
	/// ***********************************************************
	/// ***********************************************************

	void timing(int option, int ind);

	template<typename T>
	void setToAllZero (T * d_vector, int length);

	inline int findKBuckets(uint * d_bucketCount, uint * h_bucketCount, int numBuckets
		                  , uint * kVals, int numKs, uint * sums, uint * markedBuckets
		                  , int numBlocks);
    
	/// ***********************************************************
	/// ***********************************************************
	/// **** HELPER GPU FUNCTIONS-KERNELS
	/// ***********************************************************
	/// ***********************************************************
	
	template <typename T>
	__global__ void assignSmartBucket 
  			(T * d_vector, int length, int numBuckets
             , double * slopes, T * pivots, int numPivots
             , uint* d_elementToBucket , uint* d_bucketCount, int offset);
             
	__global__ void sumCounts(uint * d_bucketCount, const int numBuckets
                            , const int numBlocks);
                            
    __global__ void reindexCounts(uint * d_bucketCount, const int numBuckets
                                , const int numBlocks, uint * d_reindexCounter
                                , uint * d_markedBuckets , const int numUniqueBuckets);
                                
    template <typename T>
  	__global__ void copyElements (T* d_vector, int length, uint* elementToBucket
                                , uint * buckets, const int numBuckets, T* newArray, uint offset
                                , uint * d_bucketCount, int numTotalBuckets);
                                
    template <typename T>
  	__global__ void copyValuesInChunk (T * outputVector, T * inputVector, uint * kList
                                     , uint * kIndices, int kListCount);
                                     
                                     
	/// ***********************************************************
	/// ***********************************************************
	/// **** GENERATE PIVOTS
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
  	__global__ void enlargeIndexAndGetElements (T * in, T * list, int size);
  	
  	
  	__global__ void enlargeIndexAndGetElements (float * in, uint * out, uint * list, int size);
  	
  	template <typename T>
	void generatePivots (uint * pivots, double * slopes, uint * d_list, int sizeOfVector
			, int numPivots, int sizeOfSample, int totalSmallBuckets, uint min, uint max);
			
	template <typename T>
	void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector
		    , int numPivots, int sizeOfSample, int totalSmallBuckets, T min, T max);
		    
		    
	/// ***********************************************************
	/// ***********************************************************
	/// **** bucketMultiSelect: the main algorithm
	/// ***********************************************************
	/// ***********************************************************
  	template <typename T>
  	T bucketMultiSelect (T* d_vector, int length, uint * kVals, int numKs, T * output, int blocks
              , int threads, int numBuckets, int numPivots);
              
    template <typename T>
  	T bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, int numKs
                              , T * outputs, int blocks, int threads);                          
                                    
}

#endif
