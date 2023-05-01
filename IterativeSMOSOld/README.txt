/* Copyright 2012 Emircan Uysaler, Jeffrey Blanchard, Erik Opavsky
 * Copyright 2011 Russel Steinbach, Jeffrey Blanchard, Bradley Gordon,
 *   and Toluwaloju Alabi
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

GGMS: Grinnell GPU Multi-Selection
All original work done for this project is licensed under the Apache 2.0
license. This project is built on the GGKS project of Alabi, Blanchard, 
Gordon, and Steinbach.


 ************************************************************************
 ************************************************************************
 ********************** LIBRARIES ***************************************
 ************************************************************************
 ************************************************************************

 THRUST: a variety of thrust primitives are used throughout GGMS, so it is
 necessary to have a copy downloaded. For convenience a copy has been
 included. 


 ************************************************************************
 ************************************************************************
 **********************  bucketMultiSelect ******************************
 ************************************************************************
 ************************************************************************


 bucketMultiSelect: The source code for bucketMultiSelect is provided in 
 bucketMultiSelect.cu. It is named as we conceptually place elements into
 buckets, determine which bucket contains the desired order statistics,
 reduces the problem to only those buckets containing desired order statistics,
 and extract the desired order statistics by sorting the reduced vector.

 The details of the algorithm are provided in the following paper:
 
 Selecting Multiple Order Statistics with a Graphics Processing Unit
 Jeffrey D. Blanchard, Erik Opavsky, and Emircan Uysaler
 Submitted, 2013.
 Online [Available]: www.math.grinnell.edu/~blanchaj/Research/Research.html

 Any work which uses this algorithm, software, or derivatives of this 
 software should cite the above paper.

 Note:

 1. naivebucketMultiSelect: this algorithm is included in the package, but
 is not executed by default.  The algorithm does not utilize the kernel
 density estimator to define buckets concentrated at the density of the 
 values in the vector.  Instead, it generates equal sized buckets from
 the minimum to maximum value in the vector.  To include this function
 in tests, change the last 0 to 1 in every instance of

 algorithmsToRun[NUMBEROFALGORITHMS]= {1, 1, 0};

 that appears in the code.  You are likely to see an improved performance
 on vectors with uniformly distributed elements, and significantly degraded
 performance on vectors from other distributions.


 ************************************************************************
 ************************************************************************
 ********************** COMPILING THE CODE  *****************************
 ************************************************************************
 ************************************************************************

 There are a variety of functions which generate executables.  All executables
 can be compiled using the included MAKEFILE.  You must ensure the all path
 variables are set appropriately for your machine.  The MAKEFILE is designed
 for a Linux operating system.  For other operating systems, you must create
 your own MAKEFILE or compile the functions individually.

 To compile the functions for testing user specified problems compile 
 compareMultiSelect, analyzeMultiSelect, and realDataTests by running

 >> make all

 To compile the functions used to generate the data for the paper 
 "Selecting Multiple Order Statistics with a Graphics Processing Unit",
 run

 >> make allSMOS

 See the MAKEFILE for compiling specific files.


 ************************************************************************
 ************************************************************************
 ********************* RUNNING TESTS, GENERATING DATA *******************
 ************************************************************************
 ************************************************************************

 Idividual Testing:
 To run a user defined set of tests, utilize
 
 >> ./compareMultiSelect 

 to generate timing data compared to sorting the vector with thrust::sort. 
 The executable will prompt you for input.

 To indentify the maximum number of order statistics bucketMultiSelect
 can acquire in the time required for sort&choose, utilize

 >> ./analyzeMultiSelect

 The executable will prompt you for input.

 Tests for generating data in the paper:
 To run the tests for the paper, execute each of the files beginning with
 SMOS:
 >> ./SMOSanalyze
 >> ./SMOStimingsOSDistrAll
 >> ./SMOStimingsDistrUniform
 >> ./SMOStimingsTableData
 >> ./SMOStimingsVectorGrowth

 Running all of these programs in the current settings will take approximately
 12 hours on a Tesla C2070.  The majority of the time is spent on vectors
 larger than 2^(26).  To run a subset of the tests, simply change the ranges
 in the SMOS*.cu files prior to compilation.

 The realDataTests program takes a .csv file as input.  For example, 

 >> ./realDataTests "house_5.csv"

 Due to it's size and its source, the data is not included in the software package.
 A compressed zip file containing the data sets for the paper is available
 on Jeff Blanchard's research page.    
 www.math.grinnell.edu/~blanchaj/Research/Research.html

 Notes: 

 1. The setting for compareMultiSelect when choosing unsigned 
 integers, unsigned integers 0-4 actually generates unsigned
 integers from 0 to 100 to permit observations in line with the findings
 reported in the paper.  To generate other discrete integer ranges, simply
 alter line 54 of generateProblems.cu.

 2. When running analyzeMultiSelect, if order statistic distributions other
 than uniform or uniform random are selected, you will likely find the tests
 returning many errors when the number of order statistics exceeds n/100.
 This occurs because the memory allocated for storing the desired order 
 statistics is set at .01*n.  If you wish to test these other order statistic
 distributions, simply alter line 232 of analyzeMultiSelect.cu. 


 ************************************************************************
 ************************************************************************
 ********************* PROCESSING DATA **********************************
 ************************************************************************
 ************************************************************************


 To process the output data, you should compile readMultiSelectOutput.cpp
 by executing

 >> make ProcessData

 Then run 

 >> ./readMultiSelectOutput

 To generate the plots and tables in the paper, run

 >> readscript

 Then, open the appropriate *.m files, input the correct date(s), and 
 execute the *.m files in MATLAB.

 To see the values reported by SMOSanalyze, execute

 >> cat CR* | grep "k value"

 where we have assumed all data files in the folder are relevant.  Adding 
 more information to "CR*" will permit you to use only the relevant files.

JDB
