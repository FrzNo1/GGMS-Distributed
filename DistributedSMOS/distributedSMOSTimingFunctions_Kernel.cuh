#ifndef DISTRIBUTEDSMOSTIMINGFUNCTIONS_KERNEL_CUH
#define DISTRIBUTEDSMOSTIMINGFUNCTIONS_KERNEL_CUH

template <typename T>
void sort_CALL(T* d_vector, unsigned int length);

template <typename T>
void copyInChunk_CALL(T * outputVector, T * inputVector, uint * kList, 
					  uint kListCount, uint numElements, int blocks, 
					  int threads);

#endif
