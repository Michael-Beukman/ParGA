# PARGA - Parallel Genetic Algorithms and Simulated Annealing
This project contains some code to perform simulated annealing and genetic algorithms, serially, using MPI and using CUDA respectively.

Concrete implementations for the travelling salesman problem (TSP) and Rosenbrock function are provided.

## TLDR
First, navigate to the code directory: `cd code`.

To run the demo, simply execute `./run.sh`, or have a look at the scripts inside `./scripts/demo`. 

To run the same experiments that we ran to get our results, which take long, run `make clean all` and all the scripts inside `./scripts/experiments`.

## Run
To run this project, you will need a standard C++ compiler (e.g. `g++`), an MPI compiler (e.g. `mpicxx`) and a CUDA compiler (e.g. `nvcc`).
To build the project, simply type `make` in the `code` directory. If you want to only build a certain part of the program, you can alternatively use `make OPTION` where `OPTION` is one of `serial`, `mpi`, `cuda`. This allows you to compile only certain parts if you lack the compiler for the others.

The code was compiled and tested on Linux Ubuntu with g++ -7.5.0, OpenMPI 4 and CUDA 10. The non-cuda code was also tested and verified on OSX with clang++ 12.0.5 and OpenMPI 4.1.

### Run Modes
There are two default run modes available, demo and experiments. The experiments option should run the code in the same way we did to generate the results for the report. This is the default option.

You can then run the experiments using the scripts `run_all_exps.sh` (for MPI), `run_all_exps_serial.sh` (for serial) and `run_all_exps_cuda.sh` (for CUDA).

Alternatively you can simply run `./bin/serial`, `mpirun -np PROCS ./bin/mpi` or `./bin/cuda`.

The other option is to run a simple demo, which runs the methods on a simple problem, and prints out the results in a simple fashion. 
To perform this, you must `make clean` first, and then run `DFLAGS=-DDEMO make` to compile with demo mode active.

To run the demos, either simply run `./run.sh` or have a look at the `scripts/demo/run_demo*.sh` scripts.
The structure of the demo is `./bin/command [iterations] [scaling]`, where `iterations` is the number of iterations you want to perform in total and scaling is one of:
- "NONE" -> No scaling, all ranks will run `iterations` iterations
- "FULL" -> Full scaling, each rank will run `iterations` / num_ranks iterations
- \<integer> (e.g. i = 2) -> Each rank will run `iterations` / num_ranks * i iterations.

The scaling argument is just applicable to the `mpi` program.


To run the full demo, simply run `./run.sh`, which will compile the demo code, as well as run it.

# Method Details
## Vocabulary and Terms
In the following, as well as in the code, there are some commonly used terms:
- Individual, Solution, Genome: A single solution instance that consists of `genome_size` numbers. This is represented by an `Individual<T>` in the code.
- Population: A collection of individuals that are used in the genetic algorithm case.
- Genome Size: The size of the individual/solutions themselves. For example, when optimising the TSP problem for 10 cities, the `genome_size` is also 10, as each solution consists of a permutation of the numbers from 0-9.
## General method structure.
### Genetic Algorithm
For the genetic algorithm, we have two buffers of memory that contain the current and next populations. We first evaluated all individuals, then found the best two and used those to perform crossover to generate the next generation. Each child was also randomly mutated.

MPI used an island based model, where each rank had its own population, and they performed an independent GA for X = 10 iterations. Then the ranks exchanged information by sending the best 2, as well as other individuals between each rank.

CUDA used a global, single population model, where each thread evaluated a single individual, and bred two parents to get two children. Block 0 calculated the best and second-best individuals from one generation, which were then used in the next generation to perform crossover with.

### Simulated Annealing
Simulated annealing consisted of using an initial temperature that is proportional to the problem size, and reducing that linearly by 0.995.

At each iteration, we created a neighbouring solution by using the same mutation methods as in the GA, and swapping according to the SA rules.

For MPI, each rank performed SA independently, and exchanged the best solution every 100 iterations.

For CUDA, we performed simulated annealing on a per thread basis for S iterations, then found the best solution in the block and saved it to global memory. Finally, we found the best block, and used that as the starting point for the next iteration. Empirically, S = 1 performed well.


# File Structure
```
├── report.pdf                      -> The report detailing our methods and results.
├── code
│   ├── Makefile                    -> The makefile that actually builds the code.
│   ├── obj                         -> Temporary object directory that contains the .o files from compilation.
│   └── results                     -> CSV files that capture our results.
│   └── src                         -> Contains the actual source code.
│       ├── CUDA                    -> All CUDA code.
│       │   ├── ga                  -> Genetic algorithm specific code.
│       │   ├── main                
│       │   │   └── cuda_main.cu    -> Main runner program for the CUDA demo and experiments.
│       │   ├── problems            -> Problem specific CUDA code.
│       │   ├── sa                  -> CUDA simulated annealing.
│       │   └── utils               -> Utilities for CUDA code, including a random wrapper.
│       ├── MPI                     -> Contains all MPI specific code.
│       │   ├── ga                  -> MPI genetic algorithm.
│       │   ├── main                
│       │   │   └── mpi_main.cpp    -> Main runner program for the MPI demo and experiments.
│       │   ├── problems            -> MPI problem specific code.
│       │   ├── sa                  -> MPI simulated annealing.
│       ├── common                  -> Serial code, as well as common code across MPI, CUDA and serial.
│       │   ├── experiments         -> Contains an experiment wrapper.
│       │   ├── metrics             -> Some utility functions regarding computing metrics and saving to files.
│       │   ├── problems            -> Problem specific code, as well as the serial implementations of SA and GA.
│       │   ├── serial              -> Contains base classes for GA and SA.
│       │   ├── utils               -> Assorted utility functions.
│       │   │   ├── random          -> Random wrapper (on CPU).
│       │   └── vendor
│       │       └── random          -> Contains random number generator from https://github.com/stdfin/random.
│       ├── serial
│       │   └── main.cpp            -> Main runner program for the serial demo and experiments.
└── ds                              -> Contains data analysis python code to create plots.
    ├── ds.py
    └── plots
```
The respective main files are quite big, and full of code, but they are similar in structure. There are functions that run a single instance of a method on a problem (e.g. MPIAnnealing on TSP, or CudaGA on Rosenbrock, etc.). Most of the code is responsible for that. 

The `<method>_run_all_experiments_according_to_args` function simply runs the experiments using arguments specified via the command lines, which are `is_sa`,  `which_exp` and `is_rosen`, which determine if the method is SA or GA, which experiment to run (generally a number from 1 - 4) and whether or not to run on Rosenbrock or TSP.

The `demo` functions simply run the short demos.


I've commented `./code/src/serial/main.cpp` somewhat to explain what those function do, and it mostly carries over to the other main files.



The experiments can be performed by running the scripts inside `./scripts/experiments/*.sh`. 

**Note: These experiments take quite long, so to just see if the code runs, rather just run the demo**

# Acknowledgements
NVIDIA's common `inc` folder (which is distributed as part of CUDA samples) is found inside `./code/src/CUDA/inc`. This was used as general cuda helper functions and the original can be found at:
- [Here](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/samples.html)
- [Github](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA](https://docs.nvidia.com/cuda/cuda-samples/index.html)

Code from [stdfin-random](https://github.com/stdfin/random) can be found under `./code/src/common/vendor/random`. This was used to generate random numbers, both in serial and parallel.


In the CUDA code itself, I used function pointers, and the way to specify which function to use from the host, when launching the kernel was obtained from this [NVIDIA forum question](https://forums.developer.nvidia.com/t/how-can-i-use-device-function-pointer-in-cuda/14405) , [answer1](https://forums.developer.nvidia.com/t/how-can-i-use-device-function-pointer-in-cuda/14405/31) and [answer2](https://forums.developer.nvidia.com/t/how-can-i-use-device-function-pointer-in-cuda/14405/32)

In the code, where I used something specific, I reference it there.

The report contains some more references to sources that I used.