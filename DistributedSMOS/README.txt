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
 **********************  FILE DESCRIPTION *******************************
 ************************************************************************
 ************************************************************************

The folder contains the following files:

Distributed version of selecting multiple order statistics:
distributedSMOS.cpp
distributedSMOS_Kernel.cu

Iterative version of selecting multiple order statistics:
iterativeSMOS.cu

Sorting version of selecting multiple order statistics:
bucketMultiSelect.cu

Algorithms for generating distributions and printing information:
generateProblems.cu
printFunctions.cu

Compare Iterative and Sorting version of SMOS:
compareIterativeSMOS.cu
iterativeSMOSTimingFunctions.cu

Compare Distributed and Iterative version of SMOS:
compareDistributedSMOS.cpp
distributedSMOSTimingFunctions.cu
distributedSMOSTimingFunctions_Kernel.cu





 ************************************************************************
 ************************************************************************
 ********************** COMPILING THE CODE  *****************************
 ************************************************************************
 ************************************************************************

To begin with, make sure you can remote connect to following computers using 
ssh without entering password. You can try typing the command
>> ssh bollee
>> ssh bellman
>> ssh householder
>> ssh mccarthy
to see if you can connect to them successfully or not.

Or instead, use your own set of computers. The key here is you can ssh to other
computers without entering the password.(SSH keys are set up)


Then cd to folder DistributedSMOS


To compare Iterative and Sorting version of SMOS, firstly using one of the four
ssh command above to connect to those computers remotely. Then, using the following
command to compile the files:
>> PATH=$PATH:/usr/local/cuda/bin
>> make iterative




To compare Distributed and Iterative version of SMOS, firstly using one of the four
ssh command above to connect to those computers remotely. Then, using the following
command to compile the files:
>> PATH=$PATH:/usr/local/cuda/bin
>> make distributed




 ************************************************************************
 ************************************************************************
 ********************* RUNNING TESTS, GENERATING DATA *******************
 ************************************************************************
 ************************************************************************
 
To begin with, you have to make sure you can remote connect to following 
computers using ssh without entering password. Try typing the command
>> ssh bollee
>> ssh bellman
>> ssh householder
>> ssh mccarthy
to see if you can connect to them successfully or not.

Or instead, use your own set of computers. The key here is you can ssh to other
computers without entering the password.(SSH keys are set up)


Then cd to folder DistributedSMOS-1


To compare Iterative and Sorting version of SMOS, with user specific value, type
>> ./compareIterativeSMOS


To compare Distributed and Iterative version of SMOS, with user specific value, type
>> ssh bollee
>> mpirun -np 4 -hosts bollee,bellman,mccarthy,householder ./compareDistributedSMOS


To run tests that are generated continuously in the paper, simply type
>> ssh bollee
>> runTests



 ************************************************************************
 ************************************************************************
 ********************* PROCESSING DATA **********************************
 ************************************************************************
 ************************************************************************


 To process the output data, you should compile readMultiSelectOutput.cpp
 by executing

 >> make ProcessData

 Then run 

 >> ./readDistriSMOSoutput

 To generate the plots and tables in the paper, run

 >> readscript

 Then, open the appropriate *.m files, input the correct date(s), and 
 execute the *.m files in MATLAB.

 To see the values reported by SMOSanalyze, execute

 >> cat CR* | grep "k value"

 where we have assumed all data files in the folder are relevant.  Adding 
 more information to "CR*" will permit you to use only the relevant files.

JDB

