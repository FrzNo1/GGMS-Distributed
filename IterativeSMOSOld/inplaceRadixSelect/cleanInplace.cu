#include <thrust/scan.h>

namespace NewInplaceRadixSelect{

  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT, uint MASK0>
__global__ void getCounts(const T *d_vec,const uint size,const uint answerSoFar, uint *digitCounts,const uint mask1,const uint offset,const T max){ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i,j;
  T value;
  extern __shared__ ushort sharedCounts[];
  for(j = 0; j < 15;j++){
     sharedCounts[j * blockDim.x + threadIdx.x] = 0;
   }
  for(i = idx; i < size; i += offset){
    value = d_vec[i] & MASK0;
    if(value >= answerSoFar && (value < max)){
       sharedCounts[((value >> BIT_SHIFT) & mask1) * blockDim.x + threadIdx.x]++;
    }
   }

   for(i = 0; i <mask1 ;i++){
     if(sharedCounts[blockDim.x * i + threadIdx.x]){
       digitCounts[i * offset + idx] = sharedCounts[blockDim.x *i + threadIdx.x];
     }
   }
}

  template<typename T, uint RADIX_SIZE, uint BIT_SHIFT, uint MASK0>
__global__ void getCountsNotShared(const T *d_vec,const uint size,const uint answerSoFar, uint *digitCounts,const uint mask1,const uint offset,const T max){ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i;
  T value;
  for(i = idx; i < size; i += offset){
    value = d_vec[i] & MASK0;
    if(value >= answerSoFar && (value < max)){
      digitCounts[((value >> BIT_SHIFT) & mask1) * offset + idx]++;
    }
   }
}

template<uint RADIX_SIZE, uint BIT_SHIFT>
uint determineDigit(uint *digitCounts, uint k, uint countSize,uint numThreads, uint size, uint &numSmaller){
  uint *numLessThanOrEqualToI;
  uint i=0, smaller = 0;
  uint adjustedSize = size - numSmaller;
  numLessThanOrEqualToI = (uint *) malloc(sizeof(uint));
  k = adjustedSize - k + 1;
  thrust::device_ptr<uint>ptr(digitCounts);
  thrust::inclusive_scan(ptr, ptr + (numThreads * (1 << RADIX_SIZE)) - 1, ptr);
  while(i < (1 << RADIX_SIZE) - 1){
    cudaMemcpy(numLessThanOrEqualToI, digitCounts + (i * numThreads + (numThreads - 1)), sizeof(uint), cudaMemcpyDeviceToHost);
    if(numLessThanOrEqualToI[0] >=  k){
      numSmaller += smaller;
      return i;
    }
    smaller = numLessThanOrEqualToI[0];
    i++;
  }

  numSmaller += smaller;
  return (1<< RADIX_SIZE) - 1;
}


template<uint BIT_SHIFT>
void updateAnswerSoFar(uint &answerSoFar,uint digitValue){
  uint digitMask = digitValue << BIT_SHIFT;
  answerSoFar |= digitMask;
}

template<uint RADIX_SIZE, uint BIT_SHIFT>
void  digitPass(uint *d_vec,uint size,uint k, uint &answerSoFar, uint *digitCounts, uint blocks, uint threadsPerBlock, uint &numSmaller){
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  uint x = ((1 << RADIX_SIZE) - 1) << BIT_SHIFT;
  const uint mask0 = ~((1 << (BIT_SHIFT)) - 1);
  uint mask1 = (1 << (RADIX_SIZE)) - 1;
  uint currentDigit;
  uint countSize = (1 << RADIX_SIZE) * blocks * threadsPerBlock; 

   cudaMemset(digitCounts,0, countSize * sizeof(uint));
  // cudaEventRecord(start, 0);
   cudaEventRecord(start, 0);

  if(RADIX_SIZE <= 4){
    getCounts<uint,RADIX_SIZE, BIT_SHIFT, mask0><<<blocks, threadsPerBlock, ((1 << RADIX_SIZE) - 1) * threadsPerBlock * sizeof(ushort)>>>(d_vec, size, answerSoFar, digitCounts,mask1,threadsPerBlock * blocks, answerSoFar | x);
  }
  else{
    getCountsNotShared<uint,RADIX_SIZE, BIT_SHIFT, mask0><<<blocks, threadsPerBlock>>>(d_vec, size, answerSoFar, digitCounts,mask1,threadsPerBlock * blocks, answerSoFar | x);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start,stop);
  printf("BIT: %d  time: %f\n", BIT_SHIFT, time);
  currentDigit = determineDigit<RADIX_SIZE, BIT_SHIFT>(digitCounts, k, countSize, threadsPerBlock * blocks, size, numSmaller);
  updateAnswerSoFar<BIT_SHIFT>(answerSoFar, currentDigit);
}

template<typename T>
T countingRadixSelect(T *d_vec, uint size, uint k){



  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  uint blocks =  dp.multiProcessorCount* 2;
  uint threadsPerBlock = dp.maxThreadsPerBlock / 2;
  uint  numSmaller = 0, answerSoFar = 0;
  uint *digitCounts;
  
  uint countSize = (1 << 4) * blocks * threadsPerBlock;
  cudaMalloc(&digitCounts, countSize * sizeof(uint));

  digitPass<4,28>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<4,24>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<4,20>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<4,16>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<4,12>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<4,8>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<4,4>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  digitPass<4,0>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);

  // digitPass<8,16>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  // digitPass<8,8>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);
  // digitPass<8,0>(d_vec, size,k, answerSoFar, digitCounts, blocks, threadsPerBlock, numSmaller);

  cudaFree(digitCounts);
  return answerSoFar;
}



}
