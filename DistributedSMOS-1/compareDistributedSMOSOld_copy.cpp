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
#include <mpi.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <limits>

#include <algorithm>
#include <unistd.h>

//Include various thrust items that are used
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>

//Include other files

#include "bucketMultiselect.cuh"
#include "iterativeSMOS.cuh"
#include "distributedSMOS.hpp"

#include "distributedSMOSTimingFunctions.cuh"
#include "generateProblems.cuh"
#include "printFunctions.cuh"



#define RANK_NUM 4
#define NUMBEROFALGORITHMS 4
char* namesOfMultiselectTimingFunctions[NUMBEROFALGORITHMS] =
        {"Sort and Choose Multiselect", "Bucket Multiselect", "IterativeSMOS", "DistributedSMOS"};
#define ORDEROFDISTRIBUTEDSMOS 3

#define NUMBEROFKDISTRIBUTIONS 5

/// ***********************************************************
/// ***********************************************************
/// **** MPI Function Libraries
/// ***********************************************************
/// ***********************************************************

// TODO potential need to include this in header file

template <typename T>
void MPI_Send_CALL(T *buf, int count, 
				  int dest, int tag, MPI_Comm comm) {
	if (std::is_same<T, int>::value) {
		MPI_Send(buf, count, MPI_INT, dest, tag, comm);
	}
	else if (std::is_same<T, unsigned int>::value) {
		MPI_Send(buf, count, MPI_UNSIGNED, dest, tag, comm);
	}
	else if (std::is_same<T, float>::value) {
		MPI_Send(buf, count, MPI_FLOAT, dest, tag, comm);
	}
	else if (std::is_same<T, double>::value) {
		MPI_Send(buf, count, MPI_DOUBLE, dest, tag, comm);
	}
}

template <typename T>
void MPI_Recv_CALL(T *buf, int count, int source, int tag,
              MPI_Comm comm, MPI_Status *status) {
	if (std::is_same<T, int>::value) {
		MPI_Recv(buf, count, MPI_INT, source, tag, comm, status);
	}
	else if (std::is_same<T, unsigned int>::value) {
		MPI_Recv(buf, count, MPI_UNSIGNED, source, tag, comm, status);
	}
	else if (std::is_same<T, float>::value) {
		MPI_Recv(buf, count, MPI_FLOAT, source, tag, comm, status);
	}
	else if (std::is_same<T, double>::value) {
		MPI_Recv(buf, count, MPI_DOUBLE, source, tag, comm, status);
	}
}


/// ***********************************************************
/// ***********************************************************
/// **** MPI Message
/// ***********************************************************
/// ***********************************************************

/*
	1: host send type to each slot in main
	2. host send distributionType to each slot in main
	3. host send kDistribution to each slot in main
	4. host send startPower to each slot in main
	5. host send stopPower to each slot in main
	6. host send startK to each slot in main
	7. host send jumpK to each slot in main
	8. host send stopK to each slot in main
	9. host send testCount to each slot in main
	
	31: host send arrayOfKs to each slot in runTests
	
	41: host send runOrder to each slot in compareMultiselectAlgorithms
	42: host send h_vec_distributed to each slot in compareMultiselectAlgorithms
	43: host send h_vec_copy to each slot in compareMultiselectAlgorithms
*/




namespace CompareDistributedSMOS {
    using namespace std;
    
  	/// ***********************************************************
	/// ***********************************************************
	/// **** compareDistributedSMOS: the main algorithm
	/// ***********************************************************
	/// ***********************************************************

    template<typename T>
    void compareMultiselectAlgorithms(uint size, uint* kVals, uint numKs, uint numTests
            , uint *algorithmsToTest, uint generateType, uint kGenerateType
            , char* fileNamecsv, int rank, T* data = NULL) {     


		typedef results_t<T>* (*ptrToTimingFunction)(T*, uint, uint *, uint, int);
        typedef void (*ptrToGeneratingFunction)(T*, uint, curandGenerator_t);
        
        char* namesOfKGenerators[NUMBEROFKDISTRIBUTIONS]
            = {"Uniform Random Ks", "Uniform Ks", "Normal Random Ks", "Cluster Ks", "Sectioned Ks"};
        
        // allocate space for operations
        T *h_vec, *h_vec_copy;
        float timeArray[NUMBEROFALGORITHMS][numTests];
        T * resultsArray[NUMBEROFALGORITHMS][numTests];
        float totalTimesPerAlgorithm[NUMBEROFALGORITHMS];
        uint winnerArray[numTests];
        uint timesWon[NUMBEROFALGORITHMS];
        uint i,j,m,x;
        int runOrder[NUMBEROFALGORITHMS];
        
        // allocate space for distributedSMOS vector
        uint distributed_size = size / RANK_NUM;
        T *h_vec_distributed, *h_vec_distributed_send, *h_vec_distributed_copy;
        h_vec_distributed = (T *) malloc(distributed_size * sizeof(T));
        h_vec_distributed_copy = (T *) malloc(distributed_size * sizeof(T));
        h_vec_distributed_send = (T *) malloc(distributed_size * sizeof(T));
        
        

        unsigned long long seed;
        results_t<T> *temp;
        ofstream fileCsv;
        timeval t1;
        
        T type = 0;

        //these are the functions that can be called
        ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] =
                {&timeSortAndChooseMultiselect<T>,
                 &timeBucketMultiselect<T>,
                 &timeIterativeSMOS<T>,
                 &timeDistributedSMOS<T>};

        ptrToGeneratingFunction *arrayOfGenerators;
        char** namesOfGeneratingFunctions;
        curandGenerator_t generator;
        
        //allocate space for h_vec, and h_vec_copy
        h_vec = (T *) malloc(size * sizeof(T));
        h_vec_copy = (T *) malloc(size * sizeof(T));

		cudaThreadSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
		
		
        if (rank == 0) {
            // this is the array of names of functions that generate problems of this type,
            // ie float, double, or uint
            namesOfGeneratingFunctions = returnNamesOfGenerators<T>();
            arrayOfGenerators = (ptrToGeneratingFunction *) returnGenFunctions<T>(type);

            printf("Files will be written to %s\n", fileNamecsv);
            fileCsv.open(fileNamecsv, ios_base::app);

            //zero out the totals and times won
            bzero(totalTimesPerAlgorithm, NUMBEROFALGORITHMS * sizeof(uint));
            bzero(timesWon, NUMBEROFALGORITHMS * sizeof(uint));

            //create the random generator.
            // curandGenerator_t generator;
            srand(unsigned(time(NULL)));

            printf("The distribution is: %s\n", namesOfGeneratingFunctions[generateType]);
            printf("The k distribution is: %s\n", namesOfKGenerators[kGenerateType]);
        }

        cudaThreadSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        


        /***********************************************/
        /*********** START RUNNING TESTS ************
        /***********************************************/


        for(i = 0; i < numTests; i++) {
			
			float currentWinningTime = INFINITY;
			
            if (rank == 0) {
                //cudaDeviceReset();
                gettimeofday(&t1, NULL);
                seed = t1.tv_usec * t1.tv_sec;
                // seed = 830359020905406;
                
                // test part
                // printf("vector generater seed: %llu\n", seed);

                for (m = 0; m < NUMBEROFALGORITHMS; m++)
                    runOrder[m] = m;

                std::random_shuffle(runOrder, runOrder + NUMBEROFALGORITHMS);     // potentially not work
                fileCsv << size << "," << numKs << "," <<
                        namesOfGeneratingFunctions[generateType] << "," <<
                        namesOfKGenerators[kGenerateType] << ",";
                        
                cudaThreadSynchronize();

                curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);       // potentially not work
                curandSetPseudoRandomGeneratorSeed(generator, seed);
                printf("Running test %u of %u for size: %u and numK: %u\n", i + 1,
                       numTests, size, numKs);
                       
                cudaThreadSynchronize();

                //generate the random vector using the specified distribution
                if (data == NULL)
                    arrayOfGenerators[generateType](h_vec, size, generator);
                else
                    h_vec = data;
                    
                cudaThreadSynchronize();

                //copy the vector to h_vec_copy, which will be used to restore it later
                memcpy(h_vec_copy, h_vec, size * sizeof(T));
                
                /*
                // test part
                printf("vector: \n");
                for (int x = 0; x < 10000; x++) {
                	printf("%u, ", h_vec[x]);
                }
                
                // test part
                printf("k: \n");
                for (int x = 0; x < numKs; x++) {
                	printf("%u, ", kVals[x]);
                }
                */
                

                winnerArray[i] = 0;
                // float currentWinningTime = INFINITY;
                //run the various timing functions
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            cudaDeviceSynchronize();
            
            
            // send information to other ranks
            if (rank == 0) {
            	for (int y = 0; y < distributed_size; y++) {
            		h_vec_distributed[y] = h_vec[y];
            	}
            	
            	for (int z = 1; z < RANK_NUM; z++) {
            		MPI_Send(runOrder, NUMBEROFALGORITHMS, MPI_INT, z, 41, MPI_COMM_WORLD);
            		
            		for (int y = 0; y < distributed_size; y++) {
            			h_vec_distributed_send[y] = h_vec[y + z * distributed_size];
            		}
            		
            		MPI_Send_CALL(h_vec_distributed_send, distributed_size, z, 42, MPI_COMM_WORLD);
            		
            		// test part
            		// MPI_Send_CALL(h_vec_copy, size, z, 43, MPI_COMM_WORLD);
            	}
            }
            
            if (rank != 0) {
            	MPI_Recv(runOrder, NUMBEROFALGORITHMS, MPI_INT, 0, 41, MPI_COMM_WORLD,
					     MPI_STATUS_IGNORE);
				MPI_Recv_CALL(h_vec_distributed, distributed_size, 0, 42, MPI_COMM_WORLD,
					     MPI_STATUS_IGNORE);
					     
				// test part
				// MPI_Recv_CALL(h_vec_copy, size, 0, 43, MPI_COMM_WORLD,
					     // MPI_STATUS_IGNORE);
            
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            cudaDeviceSynchronize();
            
            /*
            //test part
            if (rank == 0) {
            	printf("from rank 0, h_vec: 0 %f, %d %f, %d %f\n", h_vec[0], size/4*2, h_vec[size/4*2], size-1, h_vec[size-1]);
            	printf("from rank 0, h_vec_distributed: 0 %f\n", h_vec_distributed[0]);
            }
            if (rank == 2) {
            	printf("from rank 2, h_vec_distributed: %d %f\n", size/4*2, h_vec_distributed[0]);
            }
            if (rank == 3) {
            	printf("from rank 3, h_vec_distributed: %d %f\n", size-1, h_vec_distributed[distributed_size-1]);
            }
            */
            
            memcpy(h_vec_distributed_copy, h_vec_distributed, distributed_size * sizeof(T));
            MPI_Barrier(MPI_COMM_WORLD);
            
            /*
            // test part
            // test whether or not vector are all same
            if (true) {
				printf("test vector on slot: distributedsize: %d\n", distributed_size);
				for (int p = 0; p < distributed_size; p++) {
					if (h_vec_copy[p + distributed_size * rank] != h_vec_distributed_copy[p]) {
						printf("rank: %d : num: %d, vec_copy: %llu, vec_distributed_copy: %llu\n", 
								rank, p, h_vec_copy[p + distributed_size * rank], h_vec_distributed_copy[p]);
					}
				}
				printf("rank %d done test vector", rank);
			}
			*/
			
            
            // run the tests for distributedSMOS
            if (algorithmsToTest[ORDEROFDISTRIBUTEDSMOS]) {
                j = ORDEROFDISTRIBUTEDSMOS;

                if (rank == 0)
                    printf("TESTING: %u\n", j);
                    
                MPI_Barrier(MPI_COMM_WORLD);

                temp = arrayOfTimingFunctions[j](h_vec_distributed_copy, distributed_size, kVals, numKs, rank);

                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 0) {
                    //record the time result
                    timeArray[j][i] = temp->time;
                    //record the value returned
                    resultsArray[j][i] = temp->vals;
                    //update the current "winner" if necessary
                    if(timeArray[j][i] < currentWinningTime){
                        currentWinningTime = temp->time;
                        winnerArray[i] = j;
                    }
                }
                
                /*
                // test part
                if (rank == 0) {
            		// test part
	                printf("testing %d:\n", j);
	                for (int xx = 0; xx < numKs; xx++) {
	                	printf("%u  ", (temp->vals)[xx]);
	                }
	                printf("\n");
                }
                */
                
				
				memcpy(h_vec_distributed_copy, h_vec_distributed, distributed_size * sizeof(T));
                free(temp);
                
                MPI_Barrier(MPI_COMM_WORLD);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);

			
            // run all the algorithms excepts the distributedSMOS
            for(x = 0; x < NUMBEROFALGORITHMS; x++){
            	MPI_Barrier(MPI_COMM_WORLD);
            	
            	
            	// skip the distributedSMOS test
            	if (runOrder[x] == ORDEROFDISTRIBUTEDSMOS)
            		continue;
            	
                j = runOrder[x];
                
                MPI_Barrier(MPI_COMM_WORLD);
                
                if(algorithmsToTest[j]){
                	if (rank == 0) {
		                //run timing function j
		                printf("TESTING: %u\n", j);
		                temp = arrayOfTimingFunctions[j](h_vec_copy, size, kVals, numKs, rank);
		                
		                cudaDeviceSynchronize();

		                //record the time result
		                timeArray[j][i] = temp->time;
		                //record the value returned
		                resultsArray[j][i] = temp->vals;
		                //update the current "winner" if necessary
		                if(timeArray[j][i] < currentWinningTime){
		                    currentWinningTime = temp->time;
		                    winnerArray[i] = j;
		                }
		                
		                /*
		                // test part
		                printf("testing %d:\n", j);
		                for (int xx = 0; xx < numKs; xx++) {
		                	printf("%u  ", (temp->vals)[xx]);
		                }
		                printf("\n");
		                */
		                

		                //perform clean up
		                free(temp);
		                memcpy(h_vec_copy, h_vec, size * sizeof(T));
                    }
                }
                
            	MPI_Barrier(MPI_COMM_WORLD);
                
            }

            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                curandDestroyGenerator(generator);
                for (x = 0; x < NUMBEROFALGORITHMS; x++)
                    if (algorithmsToTest[x])
                        fileCsv << namesOfMultiselectTimingFunctions[x] << "," << timeArray[x][i] << ",";

                // check for errors, and output information to recreate problem
                uint flag = 0;
                for (m = 1; m < NUMBEROFALGORITHMS; m++)
                    if (algorithmsToTest[m])
                        for (j = 0; j < numKs; j++) {
                            if (resultsArray[m][i][j] != resultsArray[0][i][j]) {
                                flag++;
                                fileCsv << "\nERROR ON TEST " << i << " of " << numTests << " tests!!!!!\n";
                                fileCsv << "\nERROR ON ALGORITHM " << m << "\n";
                                fileCsv << "vector size = " << size << "\nvector seed = " << seed << "\n";
                                fileCsv << "numKs = " << numKs << "\n";
                                fileCsv << "wrong k = " << kVals[j] << " kIndex = " << j <<
                                        " wrong result = " << resultsArray[m][i][j] << " correct result = " <<
                                        resultsArray[0][i][j] << "\n";
                                std::cout << namesOfMultiselectTimingFunctions[m] <<
                                          " did not return the correct answer on test " << i + 1 << " at k[" << j <<
                                          "].  It got " << resultsArray[m][i][j];
                                std::cout << " instead of " << resultsArray[0][i][j] << ".\n";
                                std::cout << "RESULT:\t";
                                PrintFunctions::printBinary(resultsArray[m][i][j]);
                                std::cout << "Right:\t";
                                PrintFunctions::printBinary(resultsArray[0][i][j]);
                                
                                // test part
                                printf("it get %.10e instead of %.10e\n", resultsArray[m][i][j], resultsArray[0][i][j]);
                            }
                            
                            cudaDeviceSynchronize();
                        }

                fileCsv << flag << "\n";
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            cudaDeviceSynchronize();
        }
        
        MPI_Barrier(MPI_COMM_WORLD);

		
        if (rank == 0) {
            //calculate the total time each algorithm took
            for (i = 0; i < numTests; i++)
                for (j = 0; j < NUMBEROFALGORITHMS; j++)
                    if (algorithmsToTest[j])
                        totalTimesPerAlgorithm[j] += timeArray[j][i];

            //count the number of times each algorithm won.
            for (i = 0; i < numTests; i++)
                timesWon[winnerArray[i]]++;

            printf("\n\n");

            //print out the average times
            for (i = 0; i < NUMBEROFALGORITHMS; i++)
                if (algorithmsToTest[i])
                    printf("%-20s averaged: %f ms\n", namesOfMultiselectTimingFunctions[i],
                           totalTimesPerAlgorithm[i] / numTests);

            for (i = 0; i < NUMBEROFALGORITHMS; i++)
                if (algorithmsToTest[i])
                    printf("%s won %u times\n", namesOfMultiselectTimingFunctions[i], timesWon[i]);

            // free results
            for (i = 0; i < numTests; i++)
                for (m = 0; m < NUMBEROFALGORITHMS; m++)
                    if (algorithmsToTest[m])
                        free(resultsArray[m][i]);


            //free h_vec and h_vec_copy
	        
	        //close the file
        	fileCsv.close();
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
    	if (data == NULL)
            free(h_vec);
        free(h_vec_copy);

        // free h_vec_distributed and h_vec_distributed_copy
    	free(h_vec_distributed);
        free(h_vec_distributed_send);
        free(h_vec_distributed_copy);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* This function generates the array of kVals to work on and acts as a wrapper for
       comparison.
    */
    template<typename T>
    void runTests (uint generateType, char* fileName, uint startPower, uint stopPower
            , uint timesToTestEachK, uint kDistribution, uint startK, uint stopK, uint kJump, int rank) {
        uint algorithmsToRun[NUMBEROFALGORITHMS]= {1, 1, 1, 1};
        uint size;
        uint i;
        uint arrayOfKs[stopK+1];
        
        typedef void (*ptrToKDistributionGenerator)(uint *, uint, uint, curandGenerator_t);
        ptrToKDistributionGenerator arrayOfKDistributionGenerators[NUMBEROFKDISTRIBUTIONS]
            = {&generateKUniformRandom, &generateKUniform, &generateKNormal, &generateKCluster, &generateKSectioned};
        
		
        // double the array size to the next powers of 2
        for(size = (1 << startPower); size <= (1 << stopPower); size *= 2) {

            if (rank == 0) {
                unsigned long long seed;
                timeval t1;
                gettimeofday(&t1, NULL);
                seed = t1.tv_usec * t1.tv_sec;
                // seed = 272017605375852;
                // test part
                // printf("k generater seed: %llu\n", seed);
                
                // write k-order seed to file
                ofstream fileCsv;
                fileCsv.open(fileName, ios_base::app);
                fileCsv << "k-order seed: " << seed << "\n";
                fileCsv.close();
                
                
                curandGenerator_t generator;
                srand(unsigned(time(NULL)));
                curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
                curandSetPseudoRandomGeneratorSeed(generator, seed);
                
                cudaThreadSynchronize();

                arrayOfKDistributionGenerators[kDistribution](arrayOfKs, stopK, size, generator);
                
                cudaThreadSynchronize();

                curandDestroyGenerator(generator);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            
            
			// potential error here: stopK is different
            if (rank == 0) {
            	for (int z = 1; z < RANK_NUM; z++) {
            		MPI_Send(arrayOfKs, stopK, MPI_UNSIGNED, z, 31, MPI_COMM_WORLD);
            	}
            }
            
            if (rank != 0) {
            	MPI_Recv(arrayOfKs, stopK, MPI_UNSIGNED, 0, 31, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	
            }
            
            
			MPI_Barrier(MPI_COMM_WORLD);
			
			/*
			//test part
            if (rank == 2) {
            	printf("from rank 2, arrayOfKs[stopK-2]: %u", arrayOfKs[stopK-2]);
            	printf("from rank 2, arrayOfKs[stopK-1]: %u", arrayOfKs[stopK-1]);
            	printf("from rank 2, arrayOfKs[stopK]: %u", arrayOfKs[stopK]);
            }
            */

            for(i = startK; i <= stopK; i+=kJump) {
                cudaDeviceReset();
                cudaThreadExit();

                if (rank == 0)
                    printf("NOW ADDING ANOTHER K\n\n");

                MPI_Barrier(MPI_COMM_WORLD);
				
				
                compareMultiselectAlgorithms<T>(size, arrayOfKs, i, timesToTestEachK,
                                                algorithmsToRun, generateType, kDistribution, fileName, rank);
                
                
                                                
                MPI_Barrier(MPI_COMM_WORLD);
                
            }

        }

    }

}



/// ***********************************************************
/// ***********************************************************
/// **** compareDistributedSMOS: the main function
/// ***********************************************************
/// ***********************************************************

int main (int argc, char *argv[]) {

    int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    uint testCount, type, distributionType,startPower,stopPower,kDistribution,startK
    ,stopK,jumpK;
    	
    // get information from user
    if (rank == 0) {
    	
    	
		printf("Please enter the type of value you want to test:\n0-float\n1-double\n2-uint\n");
		scanf("%u", &type);
		printf("Please enter distribution type: ");
		printDistributionOptions(type);
		scanf("%u", &distributionType);
		printf("Please enter K distribution type: ");
		printKDistributionOptions();
		scanf("%u", &kDistribution);
		printf("Please enter Start power: ");
		scanf("%u", &startPower);
		printf("Please enter Stop power: ");
		scanf("%u", &stopPower);
		printf("Please enter Start number of K values: ");
		scanf("%u", &startK);
		printf("Please enter number of K values to jump by: ");
		scanf("%u", &jumpK);
		printf("Please enter Stop number of K values: ");
		scanf("%u", &stopK);
		printf("Please enter number of tests to run per K: ");
		scanf("%u", &testCount);
		
		
		
		/*
		// test part
		type = 0;
		distributionType = 0;
		kDistribution = 0;
		startPower = 26;
		stopPower = 26;
		startK = 900;
		jumpK = 10;
		stopK =900;
		testCount = 4;
		*/
		
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // send distribution information to the user
    if (rank == 0) {
    	for (int i = 1; i < RANK_NUM; i++){ 
    		MPI_Send(&type, 1, MPI_UNSIGNED, i, 1, MPI_COMM_WORLD);
    		MPI_Send(&distributionType, 1, MPI_UNSIGNED, i, 2, MPI_COMM_WORLD);
    		MPI_Send(&kDistribution, 1, MPI_UNSIGNED, i, 3, MPI_COMM_WORLD);
    		MPI_Send(&startPower, 1, MPI_UNSIGNED, i, 4, MPI_COMM_WORLD);
    		MPI_Send(&stopPower, 1, MPI_UNSIGNED, i, 5, MPI_COMM_WORLD);
    		MPI_Send(&startK, 1, MPI_UNSIGNED, i, 6, MPI_COMM_WORLD);
    		MPI_Send(&jumpK, 1, MPI_UNSIGNED, i, 7, MPI_COMM_WORLD);
    		MPI_Send(&stopK, 1, MPI_UNSIGNED, i, 8, MPI_COMM_WORLD);
    		MPI_Send(&testCount, 1, MPI_UNSIGNED, i, 9, MPI_COMM_WORLD);
    	}
    }
    
    // each rank receive information
    if (rank != 0) {
    	MPI_Recv(&type, 1, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(&distributionType, 1, MPI_UNSIGNED, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(&kDistribution, 1, MPI_UNSIGNED, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(&startPower, 1, MPI_UNSIGNED, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(&stopPower, 1, MPI_UNSIGNED, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(&startK, 1, MPI_UNSIGNED, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(&jumpK, 1, MPI_UNSIGNED, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(&stopK, 1, MPI_UNSIGNED, 0, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(&testCount, 1, MPI_UNSIGNED, 0, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    /*
    // test part
    if (rank == 2) {
    	printf("rank 2, type: %u\n", type);
    	printf("rank 2, distributionType: %u\n", distributionType);
    	printf("rank 2, kDistribution: %u\n", kDistribution);
    	printf("rank 2, startPower: %u\n", startPower);
    	printf("rank 2, stopPower: %u\n", stopPower);
    	printf("rank 2, startK: %u\n", startK);
    	printf("rank 2, jumpK: %u\n", jumpK);
    	printf("rank 2, stopK: %u\n", stopK);
    	printf("rank 2, testCount: %u\n", testCount);
    }
    */
    
    

    switch(type){
        case 0:
            typeString = "float";
            break;
        case 1:
            typeString = "double";
            break;
        case 2:
            typeString = "uint";
            break;
        default:
            break;
    }

	if (rank == 0) {
		snprintf(fileName, 128,
		         "%s %s k-dist:%s 2^%d to 2^%d (%d:%d:%d) %d-tests on %s at %s",
		         typeString, getDistributionOptions(type, distributionType),
		         getKDistributionOptions(kDistribution), startPower, stopPower,
		         startK, jumpK, stopK, testCount, hostName, humanTime);
		printf("File Name: %s \n", fileName);
    }

    using namespace CompareDistributedSMOS;
    
    MPI_Barrier(MPI_COMM_WORLD);
    

	
    switch(type){
        case 0:
            runTests<float>(distributionType,fileName,startPower,stopPower,testCount,
                            kDistribution,startK,stopK,jumpK,rank);
            break;
        case 1:
            runTests<double>(distributionType,fileName,startPower,stopPower,testCount,
                             kDistribution,startK,stopK,jumpK,rank);
            break;
        case 2:
            runTests<uint>(distributionType,fileName,startPower,stopPower,testCount,
                           kDistribution,startK,stopK,jumpK,rank);
            break;
        default:
            printf("You entered and invalid option, now exiting\n");
            break;
    }
    
    

    free (fileName);
    
    
    MPI_Finalize();
    
    
    return 0;
}


