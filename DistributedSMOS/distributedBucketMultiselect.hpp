#ifndef DISTRIBUTEDBUCKETMULTISELECT_HPP
#define DISTRIBUTEDBUCKETMULTISELECT_HPP

namespace DistributedBucketMultiselect {
	template <typename T>
	T distributedBucketMultiselect 
			(T* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, T* output, int blocks, int threads, int numBuckets, int numPivots, 
			 int rank);
			 
	template <typename T>
	T distributedBucketMultiselectWrapper (T* d_vector_local, int length_local, 
			unsigned int * kVals_ori, int numKs, T* output, int blocks, int threads, 
			int rank);

}

#endif
