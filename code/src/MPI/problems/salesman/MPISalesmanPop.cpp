#include "MPI/problems/salesman/MPISalesmanPop.h"
#include <mpi.h>
#include "common/utils/random/Random.h"
#include "common/utils/utils.h"
void MPISalesmanPop::solveProblem(int num_gens)
{

    // first send data
    MPI_Bcast(mem_pop, genome_size * population_size, MPI_INT, root, MPI_COMM_WORLD);
    // MPI_Bcast(mem_next_pop, genome_size * population_size, MPI_INT, root, MPI_COMM_WORLD);
    memcpy(mem_next_pop, mem_pop ,genome_size * population_size * sizeof(int));

    // set pop data.
    pop = mem_pop;
    next_pop = mem_next_pop;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    pop_per_rank = ceil((double)population_size / num_ranks);
    start = pop_per_rank * myrank;
    end = std::min(pop_per_rank * (myrank + 1), population_size);

    for (int gen_count = 0; gen_count <= num_gens; ++gen_count)
    {

        // then evaluate on each process
        float local_max = evaluate();

        if ((gen_count + 1) % copy_interval == 0)
        {
            float recvbuf;
            MPI_Reduce(&local_max, &recvbuf, 1, MPI_FLOAT, MPI_MAX, root, MPI_COMM_WORLD);
            if (myrank == 0)
            {
                all_scores.push_back(1 / recvbuf);
            }
        }
        float global_total = local_total;
        for (int i = 0; i < population_size; ++i)
        {
            probabilities[i] /= global_total;
        }

        breed(0, population_size, true, start);

        if ((gen_count + 1) % copy_interval == 0)
        {
            // exchange data.
            MPI_Allgather(MPI_IN_PLACE, (end - start) * genome_size, MPI_INT, mem_next_pop, (end - start) * genome_size, MPI_INT, MPI_COMM_WORLD);
            MPI_Allgather(MPI_IN_PLACE, (end - start) * genome_size, MPI_INT, mem_pop, (end - start) * genome_size, MPI_INT, MPI_COMM_WORLD);
        }
        std::swap(pop, next_pop);
    }
}

float MPISalesmanPop::evaluate()
{

    local_total = 0.0f;
    float maxVal = 0;

    for (int i = 0; i < population_size; ++i)
    {
        probabilities[i] = evaluateSingle(getIndividual(i, pop));
        local_total += probabilities[i];
        maxVal = std::max(maxVal, probabilities[i]);
    }
    return maxVal;
}
