#include "MPI/ga/MPIPopulation.h"

#include <mpi.h>

template <typename T>
void MPIPopulation<T>::init() {
    // normal init
    serial_class->init();

    // and then the number of ranks and myrank.
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
}

/**
 * @brief Returns the MPI_Datatype of a template argument.
 *          from here: https://stackoverflow.com/a/42615098
 * 
 */
template <typename T>
MPI_Datatype get_type() {
    char name = typeid(T).name()[0];
    switch (name) {
        case 'i':
            return MPI_INT;
        case 'f':
            return MPI_FLOAT;
        case 'j':
            return MPI_UNSIGNED;
        case 'd':
            return MPI_DOUBLE;
        case 'c':
            return MPI_CHAR;
        case 's':
            return MPI_SHORT;
        case 'l':
            return MPI_LONG;
        case 'm':
            return MPI_UNSIGNED_LONG;
        case 'b':
            return MPI_BYTE;
    }
    return MPI_INT; // as default.
}

template <typename T>
void MPIPopulation<T>::solveProblem(int num_gens) {
    T* mem_pop = serial_class->mem_pop;
    T* mem_next_pop = serial_class->mem_next_pop;
    int genome_size = serial_class->genome_size;
    int population_size = serial_class->population_size;
    MPI_Datatype INDIV_TYPE = get_type<T>(); // type can be multiple different things, like int or float.

    // broadcast starting point to all ranks.
    MPI_Bcast(mem_pop, genome_size * population_size, INDIV_TYPE, root, MPI_COMM_WORLD);
    // Setup initial next pop.
    memcpy(mem_next_pop, mem_pop, genome_size * population_size * sizeof(T));

    serial_class->pop = mem_pop;
    serial_class->next_pop = mem_next_pop;

    pop_per_rank = ceil((double)population_size / num_ranks);
    start = pop_per_rank * myrank;
    end = std::min(pop_per_rank * (myrank + 1), population_size);

    for (int gen_count = 0; gen_count <= num_gens; ++gen_count) {
        // then evaluate on each process
        float local_max = evaluate();

        // Only communicate every copy_interval generations
        if ((gen_count + 1) % copy_interval == 0) {
            float recvbuf;
            // reduce and find the best score, to be able to write it to the results.
            MPI_Reduce(&local_max, &recvbuf, 1, MPI_FLOAT, MPI_MAX, root, MPI_COMM_WORLD);
            if (myrank == 0) {
#ifdef MPI_VERBOSE
                printf("MPI GA. Rank = 0, localtotal = %f, Global Max = %f\n", local_max, recvbuf);
#endif
                serial_class->all_scores.push_back(1 / recvbuf);
            }
        }

        float global_total = local_total;
        // normalise probabilities.
        for (int i = 0; i < population_size; ++i) {
            serial_class->probabilities[i] /= global_total;
        }

        // Breed the entire population, and save best 2 indivs to position start and start + 1.
        serial_class->breed(0, population_size, true, start);

        if ((gen_count + 1) % copy_interval == 0) {
            // exchange data, use INDIV_TYPE, as it can be either float or int.
            // We exchange this pop and next pop, and we send the best 2 indivs, and a few others from each rank to every other rank.
            // If we have 3 ranks, with arrays:
            // rank 1: a b c d e f
            // rank 2: A B C D E F
            // rank 3: 1 2 3 4 5 6
            
            // then this all reduce will result in everyone having:
            // a b C D 5 6
            MPI_Allgather(MPI_IN_PLACE, (end - start) * genome_size, INDIV_TYPE, mem_next_pop, (end - start) * genome_size, INDIV_TYPE, MPI_COMM_WORLD);
            MPI_Allgather(MPI_IN_PLACE, (end - start) * genome_size, INDIV_TYPE, mem_pop, (end - start) * genome_size, INDIV_TYPE, MPI_COMM_WORLD);
        }
        std::swap(serial_class->pop, serial_class->next_pop);
    }
}

template <typename T>
float MPIPopulation<T>::evaluate() {
    local_total = 0.0f;
    float maxVal = 0;

    // Evaluates whole population.
    for (int i = 0; i < serial_class->population_size; ++i) {
        serial_class->probabilities[i] = serial_class->evaluateSingle(serial_class->getIndividual(i, serial_class->pop));
        local_total += serial_class->probabilities[i];
        maxVal = std::max(maxVal, serial_class->probabilities[i]);
    }
    return maxVal;
}

template class MPIPopulation<int>;
template class MPIPopulation<float>;