#ifndef __MPIMASTERSLAVEPOP_H__
#define __MPIMASTERSLAVEPOP_H__

#include "common/serial/Population.h"
#include "common/problems/salesman/TSP.h"
#include <mpi.h>

/**
 * @brief A master slave MPI genetic algorithm. This wasn't really in the report, as it had quite limited scaling.
 * It is also quite hacky at the moment.
 * 
 */
class MPIMasterSlavePop{
    public:
    int rank, numranks;
    const TSP& prob;
    int pop_size;
    int genome_size;
    MPIMasterSlavePop(int _popsize, const TSP& _prob): pop_size(_popsize), prob(_prob), genome_size(_prob.C){}
    int* pop = nullptr;
    int* next_pop = nullptr;
    float* fitnesses = nullptr;
    MPI_Request* array_of_requests = nullptr;
    int* indices = nullptr;
    float max_fitness, total_fitness;
    void init();
    void solveProblem(int num_gens);
    ~MPIMasterSlavePop();
};
#endif // __MPIMASTERSLAVEPOP_H__