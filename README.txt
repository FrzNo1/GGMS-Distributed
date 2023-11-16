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
ssh bollee
ssh bellman
ssh householder
ssh mccarthy
to see if you can connect to them successfully or not.


Then cd to folder DistributedSMOS


To compare Iterative and Sorting version of SMOS, firstly using one of the four
ssh command above to connect to those computers remotely. Then, using the following
command one by one to compile the files:
PATH=$PATH:/usr/local/cuda/bin
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c iterativeSMOS.cu -o iterativeSMOS.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c bucketMultiselect.cu -o bucketMultiselect.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c iterativeSMOSTimingFunctions.cu -o iterativeSMOSTimingFunctions.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c generateProblems.cu -o generateProblems.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c printFunctions.cu -o printFunctions.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c compareIterativeSMOS.cu -o compareIterativeSMOS.o
nvcc iterativeSMOS.o bucketMultiselect.o iterativeSMOSTimingFunctions.o generateProblems.o printFunctions.o compareIterativeSMOS.o -lcudart -lcurand -I./lib/ -I. -L/usr/local/cuda/lib64 -arch=sm_60 -o compareIterativeSMOS




To compare Distributed and Iterative version of SMOS, firstly using one of the four
ssh command above to connect to those computers remotely. Then, using the following
command one by one to compile the files:
PATH=$PATH:/usr/local/cuda/bin
mpic++ -c -I/usr/local/cuda/include -I/usr/local/cuda-11.4/include -I./lib/ -I. -L/usr/local/cuda/lib64 distributedSMOS.cpp -o distributedSMOS.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c distributedSMOS_Kernel.cu -o distributedSMOS_Kernel.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c iterativeSMOS.cu -o iterativeSMOS.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c distributedSMOSTimingFunctions_Kernel.cu -o distributedSMOSTimingFunctions_Kernel.o
mpic++ -c -I/usr/local/cuda/include -I/usr/local/cuda-11.4/include -I./lib/ -I. -L/usr/local/cuda/lib64 distributedSMOSTimingFunctions.cpp -o distributedSMOSTimingFunctions.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c generateProblems.cu -o generateProblems.o
nvcc -I./lib/ -I. -arch=sm_60 -lcurand -lm -c printFunctions.cu -o printFunctions.o
mpic++ -c -I/usr/local/cuda/include -I/usr/local/cuda-11.4/include -I./lib/ -I. -L/usr/local/cuda/lib64 compareDistributedSMOS.cpp -o compareDistributedSMOS.o
mpic++ compareDistributedSMOS.o distributedSMOSTimingFunctions.o distributedSMOSTimingFunctions_Kernel.o generateProblems.o printFunctions.o bucketMultiselect.o iterativeSMOS.o distributedSMOS.o distributedSMOS_Kernel.o -lcudart -lcurand -I./lib/ -I. -L/usr/local/cuda/lib64 -o compareDistributedSMOS




 ************************************************************************
 ************************************************************************
 ********************* RUNNING TESTS, GENERATING DATA *******************
 ************************************************************************
 ************************************************************************
 
To begin with, you have to make sure you can remote connect to following 
computers using ssh without entering password. Try typing the command
ssh bollee
ssh bellman
ssh householder
ssh mccarthy
to see if you can connect to them successfully or not.


Then cd to folder DistributedSMOS-1


To compare Iterative and Sorting version of SMOS, type
./compareIterativeSMOS


To compare Distributed and Iterative version of SMOS, type
mpirun -np 4 -hosts bollee,bellman,mccarthy,householder ./compareDistributedSMOS


Github Token: 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
