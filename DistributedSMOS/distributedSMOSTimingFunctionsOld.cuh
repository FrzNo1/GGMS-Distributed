#ifndef DISTRIBUTEDSMOSTIMINGFUNCTIONSOLD_HPP
#define DISTRIBUTEDSMOSTIMINGFUNCTIONSOLD_HPP

template <typename T>
struct results_t {
    float time;
    T * vals;
};

template <typename T>
void setupForTiming(cudaEvent_t &start, cudaEvent_t &stop, T * h_vec, T ** d_vec,
                    results_t<T> ** result, uint numElements, uint kCount);
                    
template <typename T>
void wrapupForTiming(cudaEvent_t &start, cudaEvent_t &stop, float time, results_t<T> * result);

template <typename T>
__global__ void copyInChunk(T * outputVector, T * inputVector, uint * kList, 
							uint kListCount, uint numElements);
							

template<typename T>
results_t<T>* timeSortAndChooseMultiselect
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
	
template<typename T>
results_t<T>* timeBucketMultiselect 
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
	
template<typename T>
results_t<T>* timeNaiveBucketMultiselect 
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
	
template<typename T>
results_t<T>* timeIterativeSMOS 
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank);
	
template<typename T>
results_t<T>* timeDistributedSMOS 
	(T * h_vec, uint numElements, uint * kVals, uint kCount, int rank);

template<typename T>
results_t<T>* timeSortAndChooseTopkselect
	(T * h_vec, uint numElements, uint kCount);
	
/*
template<typename T>
results_t<T>* timeRandomizedTopkselect 
	(T * h_vec, uint numElements, uint kCount);
*/
	
template<typename T>
results_t<T>* timeBucketTopkselect 
	(T * h_vec, uint numElements, uint kCount);
	

#endif
