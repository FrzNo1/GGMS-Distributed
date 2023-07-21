#ifndef DISTRIBUTEDSMOS_HPP
#define DISTRIBUTEDSMOS_HPP

// TODO: potential need to include this in hpp file
/*
template <typename T>
void MPI_Send_CALL(T *buf, int count, 
				  int dest, int tag, MPI_Comm comm);
				  
template <typename T>
void MPI_Recv_CALL(T *buf, int count, int source, int tag,
              MPI_Comm comm, MPI_Status *status);
*/

namespace DistributedSMOS {
template <typename T>
	T distributedSMOS 
			(T* d_vector_local, int length_local, int length_total, unsigned int * kVals, 
			 int numKs, T* output, int blocks, int threads, int numBuckets, int numPivots, 
			 int rank);
			 
	template <typename T>
	T distributedSMOSWrapper 
			(T* d_vector_local, int length_local, unsigned int * kVals_ori, 
			 int numKs, T* output, int blocks, int threads, int rank);
}

#endif
