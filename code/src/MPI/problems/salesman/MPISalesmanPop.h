#ifndef __MPISALESMANPOP_H__
#define __MPISALESMANPOP_H__

#include "common/problems/salesman/ga/SalesmanPop.h"

/**
 * @brief MPI Salesman population.
 * @deprecated  This is actually an old version, the new one uses the serial population class, and is found in 
 * `src/MPI/ga/MPIPopulation.h` This is just here for completeness.
 */
class MPISalesmanPop : public SalesmanPopulation {
    const int root = 0;
    int myrank, num_ranks, pop_per_rank, start, end;
    float local_total = 0.0f;
    int copy_interval = 10;
   public:
   MPISalesmanPop(int pop_size, int _genome_size, TSP& problem) : SalesmanPopulation(pop_size, _genome_size, problem){}
    void solveProblem(int num_gens) override;
    float evaluate() override;

};
#endif  // __MPISALESMANPOP_H__