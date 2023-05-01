/* Copyright 2012 Jeffrey Blanchard, Erik Opavsky, and Emircan Uysaler
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sys/time.h>

#include <algorithm>
//Include various thrust items that are used
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>

//various functions, include the functions
//that print numbers in binary.
#include "printFunctions.cu"

//the algorithms
#include "bucketMultiselect.cu"
//#include "naiveBucketMultiselect.cu"

#include "generateProblems.cu"
#include "multiselectTimingFunctions.cu"

#define NUMBEROFALGORITHMS 2
char* namesOfMultiselectTimingFunctions[NUMBEROFALGORITHMS] = 
{"Sort and Choose Multiselect", "Bucket Multiselect"};

using namespace std;

namespace CompareMultiselect {

  /* This function compares bucketMultiselect with the other algorithms given in the
     defined range of kVals and array size.
  */
template<typename T>
void compareMultiselectAlgorithms(uint size, uint* kVals, uint numKs, uint numTests
, uint *algorithmsToTest, uint generateType, uint kGenerateType, char* fileNamecsv
, T* data = NULL) {

  // allocate space for operations
  T *h_vec, *h_vec_copy;
  float timeArray[NUMBEROFALGORITHMS][numTests];
  T * resultsArray[NUMBEROFALGORITHMS][numTests];
  float totalTimesPerAlgorithm[NUMBEROFALGORITHMS];
  uint winnerArray[numTests];
  uint timesWon[NUMBEROFALGORITHMS];
  uint i,j,m,x;
  int runOrder[NUMBEROFALGORITHMS];

  unsigned long long seed; //, seed2;
  results_t<T> *temp;
  ofstream fileCsv;
  timeval t1; //, t2;
 
  typedef results_t<T>* (*ptrToTimingFunction)(T*, uint, uint *, uint);
  typedef void (*ptrToGeneratingFunction)(T*, uint, curandGenerator_t);

  //these are the functions that can be called
  ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] = 
    {&timeSortAndChooseMultiselect<T>,
     &timeBucketMultiselect<T>};
  
  ptrToGeneratingFunction *arrayOfGenerators;
  char** namesOfGeneratingFunctions;
  
  // this is the array of names of functions that generate problems of this type, 
  // ie float, double, or uint
  namesOfGeneratingFunctions = returnNamesOfGenerators<T>();
  arrayOfGenerators = (ptrToGeneratingFunction *) returnGenFunctions<T>();

  printf("Files will be written to %s\n", fileNamecsv);
  fileCsv.open(fileNamecsv, ios_base::app);
  
  //zero out the totals and times won
  bzero(totalTimesPerAlgorithm, NUMBEROFALGORITHMS * sizeof(uint));
  bzero(timesWon, NUMBEROFALGORITHMS * sizeof(uint));

  //allocate space for h_vec, and h_vec_copy
  h_vec = (T *) malloc(size * sizeof(T));
  h_vec_copy = (T *) malloc(size * sizeof(T));

  //create the random generators.
  curandGenerator_t generator;
  srand(unsigned(time(NULL)));

  printf("The distribution is: %s\n", namesOfGeneratingFunctions[generateType]);
  printf("The k distribution is: %s\n", namesOfKGenerators[kGenerateType]);

  /***********************************************/
  /*********** START RUNNING TESTS ************
  /***********************************************/

  for(i = 0; i < numTests; i++) {
    //cudaDeviceReset();
    gettimeofday(&t1, NULL);
    seed = t1.tv_usec * t1.tv_sec;
    
    for(m = 0; m < NUMBEROFALGORITHMS;m++)
      runOrder[m] = m;
    
    std::random_shuffle(runOrder, runOrder + NUMBEROFALGORITHMS);
    fileCsv << size << "," << numKs << "," << 
      namesOfGeneratingFunctions[generateType] << "," << 
      namesOfKGenerators[kGenerateType] << ",";

    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,seed);
    printf("Running test %u of %u for size: %u and numK: %u\n", i + 1, 
           numTests, size, numKs);

    //generate the random vector using the specified distribution
    if(data == NULL) 
      arrayOfGenerators[generateType](h_vec, size, generator);
    else
      h_vec = data;

    //copy the vector to h_vec_copy, which will be used to restore it later
    memcpy(h_vec_copy, h_vec, size * sizeof(T));

/*
***************************************************
****** In this file, the kDistribution is always set to UNIFORM (kGenerateType = 1)
****** so this regeneration of the order statistics is not needed.
****** It is saved here in case one wants to run these tests for a different kDistribution
***************************************************
    // if the kdistribution is random, we need to generate new a kList for each new random problem instance.
    if ( (kGenerateType != 1) && (i>0) ){
      gettimeofday(&t2, NULL);
      seed2 = t2.tv_usec * t2.tv_sec;
      curandGenerator_t generator2;
      srand(unsigned(time(NULL)));
      curandCreateGenerator(&generator2, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(generator2,seed2);

      arrayOfKDistributionGenerators[kGenerateType](kVals, numKs, size, generator2);
    }
*/


    winnerArray[i] = 0;
    float currentWinningTime = INFINITY;
    //run the various timing functions
    for(x = 0; x < NUMBEROFALGORITHMS; x++){
      j = runOrder[x];
      if(algorithmsToTest[j]){

        //run timing function j
        printf("TESTING: %u\n", j);
        temp = arrayOfTimingFunctions[j](h_vec_copy, size, kVals, numKs);

        //record the time result
        timeArray[j][i] = temp->time;
        //record the value returned
        resultsArray[j][i] = temp->vals;
        //update the current "winner" if necessary
        if(timeArray[j][i] < currentWinningTime){
          currentWinningTime = temp->time;
          winnerArray[i] = j;
        }

        //perform clean up 
        free(temp);
        memcpy(h_vec_copy, h_vec, size * sizeof(T));
      }
    }

    curandDestroyGenerator(generator);
    for(x = 0; x < NUMBEROFALGORITHMS; x++)
      if(algorithmsToTest[x])
        fileCsv << namesOfMultiselectTimingFunctions[x] << "," << timeArray[x][i] << ",";

    // check for errors, and output information to recreate problem
    uint flag = 0;
    for(m = 1; m < NUMBEROFALGORITHMS;m++)
      if(algorithmsToTest[m])
        for (j = 0; j < numKs; j++) {
          if(resultsArray[m][i][j] != resultsArray[0][i][j]) {
            flag++;
            fileCsv << "\nERROR ON TEST " << i << " of " << numTests << " tests!!!!!\n";
            fileCsv << "vector size = " << size << "\nvector seed = " << seed << "\n";
            fileCsv << "numKs = " << numKs << "\n";
            fileCsv << "wrong k = " << kVals[j] << " kIndex = " << j << 
              " wrong result = " << resultsArray[m][i][j] << " correct result = " <<  
              resultsArray[0][i][j] << "\n";
            std::cout <<namesOfMultiselectTimingFunctions[m] <<
              " did not return the correct answer on test " << i + 1 << " at k[" << j << 
              "].  It got "<< resultsArray[m][i][j];
            std::cout << " instead of " << resultsArray[0][i][j] << ".\n" ;
            std::cout << "RESULT:\t";
            PrintFunctions::printBinary(resultsArray[m][i][j]);
            std::cout << "Right:\t";
            PrintFunctions::printBinary(resultsArray[0][i][j]);
          }
        }

    fileCsv << flag << "\n";
  }
  
  //calculate the total time each algorithm took
  for(i = 0; i < numTests; i++)
    for(j = 0; j < NUMBEROFALGORITHMS;j++)
      if(algorithmsToTest[j])
        totalTimesPerAlgorithm[j] += timeArray[j][i];

  //count the number of times each algorithm won. 
  for(i = 0; i < numTests;i++)
    timesWon[winnerArray[i]]++;

  printf("\n\n");

  //print out the average times
  for(i = 0; i < NUMBEROFALGORITHMS; i++)
    if(algorithmsToTest[i])
      printf("%-20s averaged: %f ms\n", namesOfMultiselectTimingFunctions[i], totalTimesPerAlgorithm[i] / numTests);

  for(i = 0; i < NUMBEROFALGORITHMS; i++)
    if(algorithmsToTest[i])
      printf("%s won %u times\n", namesOfMultiselectTimingFunctions[i], timesWon[i]);

  // free results
  for(i = 0; i < numTests; i++) 
    for(m = 0; m < NUMBEROFALGORITHMS; m++) 
      if(algorithmsToTest[m])
        free(resultsArray[m][i]);

  //free h_vec and h_vec_copy
  if(data == NULL) 
    free(h_vec);
  free(h_vec_copy);

  //close the file
  fileCsv.close();
}

  /* This function generates the array of kVals to work on and acts as a wrapper for 
     comparison.
   */
template<typename T>
void runTests (uint generateType, char* fileName, uint startPower, uint stopPower
, uint timesToTestEachK, uint kDistribution, uint startK, uint stopK, uint kJump) {
  uint algorithmsToRun[NUMBEROFALGORITHMS]= {1, 1};
  uint size;
//  uint i;
  uint arrayOfKs[stopK+1];

  /*
  *****************************
  **** In this file, the kDistribution is not random.  
  **** The number of order statistics (numKs) is fixed at 101.
  **** We only need to generate the kDistribuion one time for each size.
  *****************************
  */
  unsigned long long seed;
  timeval t1;
  gettimeofday(&t1, NULL);
  seed = t1.tv_usec * t1.tv_sec;
  curandGenerator_t generator;
  srand(unsigned(time(NULL)));
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator,seed);
  
  // double the array size to the next powers of 2
  for(size = (1 << startPower); size <= (1 << stopPower); size *= 2) {

    cudaDeviceReset();
    cudaThreadExit();

    arrayOfKDistributionGenerators[kDistribution](arrayOfKs, stopK, size, generator);

    compareMultiselectAlgorithms<T>(size, arrayOfKs, stopK, timesToTestEachK, 
                                      algorithmsToRun, generateType, kDistribution, fileName);
                                
  } // end for(size)
  curandDestroyGenerator(generator);
} // end runTests

} // end namespace CompareMultiselect


int main (int argc, char *argv[]) {

  using namespace CompareMultiselect;

  char *fileName, *hostName, *typeString;

  fileName = (char*) malloc(128 * sizeof(char));
  typeString = (char*) malloc(10 * sizeof(char));
  hostName = (char*) malloc(20 * sizeof(char));
  gethostname(hostName, 20);

  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  char * humanTime = asctime(timeinfo);
  humanTime[strlen(humanTime)-1] = '\0';

  uint testCount, type,distributionType,startPower,stopPower,kDistribution,startK
    ,stopK,jumpK;
  uint vecDistr[4];

  vecDistr[0]=0;  // Uniform
  vecDistr[1]=1;  // Normal
  vecDistr[2]=3;  // Half Normal
  vecDistr[3]=9;  // Cauchy


  kDistribution=1;  // Uniformly Spaced
  startPower=20;
  stopPower=28;
  startK=101;       // This gives the 0,1,2,...,98,99,100 percentiles
  jumpK=1;
  stopK=101;
  testCount=25;
  

  for(int j=0; j<4; j++){
    distributionType = vecDistr[j];
    for(type=0; type<3; type++){

      switch(type){
      case 0:
        typeString = "float";
        snprintf(fileName, 128, 
               "%s %s k-dist:%s 2^%d to 2^%d (%d:%d:%d) %d-tests on %s at %s", 
               typeString, getDistributionOptions(type, distributionType), 
               getKDistributionOptions(kDistribution), startPower, stopPower, 
               startK, jumpK, stopK, testCount, hostName, humanTime);
        printf("File Name: %s \n", fileName);
        runTests<float>(distributionType,fileName,startPower,stopPower,testCount,
                        kDistribution,startK,stopK,jumpK);
        break;
      case 1:
        typeString = "double";
        if (distributionType<2){
          snprintf(fileName, 128, 
               "%s %s k-dist:%s 2^%d to 2^%d (%d:%d:%d) %d-tests on %s at %s", 
               typeString, getDistributionOptions(type, distributionType), 
               getKDistributionOptions(kDistribution), startPower, stopPower, 
               startK, jumpK, stopK, testCount, hostName, humanTime);
          printf("File Name: %s \n", fileName);
          runTests<double>(distributionType,fileName,startPower,stopPower,testCount,
                         kDistribution,startK,stopK,jumpK);
        } // end if(distributionType)
        break;
      case 2:
        typeString = "uint";
        if (distributionType<1){
          snprintf(fileName, 128, 
               "%s %s k-dist:%s 2^%d to 2^%d (%d:%d:%d) %d-tests on %s at %s", 
               typeString, getDistributionOptions(type, distributionType), 
               getKDistributionOptions(kDistribution), startPower, stopPower, 
               startK, jumpK, stopK, testCount, hostName, humanTime);
          printf("File Name: %s \n", fileName);
          runTests<uint>(distributionType,fileName,startPower,stopPower,testCount,
                       kDistribution,startK,stopK,jumpK);
        } // end if(distributionType)
        break;
      default:
        printf("You entered and invalid option, now exiting\n");
        break;
      } // end switch(type)

    } // end for(type)
  } // end for (int j)


  free (fileName);
  return 0;
}

