#ifndef PRINTFUNCTIONS_CUH
#define PRINTFUNCTIONS_CUH

namespace PrintFunctions{

	union udub{
	double d;
	unsigned long long ull;
	};
	union uf{
	float f;
	unsigned int u;
	};
	
	template<typename T>
  	void printArray(T *h_vec,uint size);

	void printArray(char **h_vec,uint size);
	
	template<typename T>
 	void printCudaArray(T *d_vec,uint size);
 	
 	void printBinary(uint input);
 	
 	void printBinary(float input);
 	
 	void printBinary(double input);
 	
 	void printBinary(unsigned long long input);
 	
 	template<typename T>
  	void printArrayBinary(T *h_vec,uint size);
  	
  	template<typename T>
  	void printCudaArrayBinary(T *d_vec,uint size);
}



#endif
