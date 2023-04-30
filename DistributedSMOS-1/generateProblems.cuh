#ifndef GENERATEPROBLEMS_CUH
#define GENERATEPROBLEMS_CUH

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE UINTS
///////////////////////////////////////////////////////////////////

typedef void (*ptrToUintGeneratingFunction)(uint*, uint, curandGenerator_t);

void generateUniformUnsignedIntegers(uint *h_vec, uint numElements, curandGenerator_t generator);

void generateSortedArrayUints(uint* input, uint length, curandGenerator_t gen);

void generateUniformZeroToFourUints (uint* input, uint length, curandGenerator_t gen);



///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE FLOATS
///////////////////////////////////////////////////////////////////
typedef void (*ptrToFloatGeneratingFunction)(float*, uint, curandGenerator_t);

void generateUniformFloats(float *h_vec, uint numElements, curandGenerator_t generator);

void generateNormalFloats(float* h_vec, uint numElements, curandGenerator_t generator);

void generateOnesTwosNoisyFloats(float* input, int length, int firstVal, int firstPercent,
                                 int secondVal, int secondPercent);
                                 
void generateOnesTwosFloats(float* input, uint length, curandGenerator_t gen);

void generateAllOnesFloats(float* input, uint length, curandGenerator_t gen);

void generateCauchyFloats(float* input, uint length, curandGenerator_t gen);

void generateNoisyVector(float* input, uint length, curandGenerator_t gen);

struct multiplyByMillion
{
    __host__ __device__
    void operator()(float &key){
        key = key * 1000000;
    }
};

void generateHugeUniformFloats(float* input, uint length, curandGenerator_t gen);

void generateNormalFloats100(float* input, uint length, curandGenerator_t gen);

void generateHalfNormalFloats(float* input, uint length, curandGenerator_t gen);

struct makeSmallFloat
{
    __host__ __device__
    void operator()(uint &key){
        key = key & 0x80EFFFFF;
    }
};

void generateBucketKillerFloats(float *h_vec, uint numElements, curandGenerator_t generator);

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE DOUBLES
///////////////////////////////////////////////////////////////////

typedef void (*ptrToDoubleGeneratingFunction)(double*, uint, curandGenerator_t);

void generateUniformDoubles(double *h_vec, uint numElements, curandGenerator_t generator);

void generateNormalDoubles(double* h_vec, uint numElements, curandGenerator_t gen);

struct makeSmallDouble
{
    __host__ __device__
    void operator()(unsigned long long &key){
        key = key & 0x800FFFFFFFFFFFFF;
    }
};

void generateBucketKillerDoubles(double *h_vec, uint numElements, curandGenerator_t generator);

template <typename T> 
void* returnGenFunctions(T type);

template<typename T> 
char** returnNamesOfGenerators();

void printDistributionOptions(uint type);

char * getDistributionOptions(uint type, uint number);

/********** K DISTRIBUTION GENERATOR FUNCTIONS ************/

void generateKUniformRandom 
	(uint * kList, uint kListCount, uint vectorSize, curandGenerator_t generator);
	
void generateKUniform 
	(uint * kList, uint kListCount, uint vectorSize, curandGenerator_t generator);
	
void generateKNormal 
	(uint * kList, uint kListCount, uint vectorSize, curandGenerator_t generator);

void generateKCluster 
	(uint * kList, uint kListCount, uint vectorSize, curandGenerator_t generator);

void generateKSectioned 
	(uint * kList, uint kListCount, uint vectorSize, curandGenerator_t generator);
	
void generateKSectioned (uint * kList, uint kListCount, uint vectorSize, curandGenerator_t generator);

typedef void (*ptrToKDistributionGenerator)(uint *, uint, uint, curandGenerator_t);

void printKDistributionOptions();

char * getKDistributionOptions(uint number);






















#endif
