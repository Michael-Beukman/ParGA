#include "MPI/ga/master_slave/MPIMasterSlavePop.h"

#include <mpi.h>
#include <omp.h>

#include <algorithm>

#include "common/problems/salesman/SalesmanUtils.h"
#include "common/utils/random/Random.h"
void salesmanCrossover(const Individual<int> parent1, const Individual<int> parent2,
                       Individual<int> child1, Individual<int> child2, int genome_size) {
    int i1 = Random::random0n(genome_size);
    int i2 = Random::random0n(genome_size);

    // v[i] is true if child1 contains city i.
    std::vector<bool> childOneContains(genome_size, 0);
    std::vector<bool> childTwoContains(genome_size, 0);

    if (i1 > i2) {
        std::swap(i1, i2);
    }

    int indexC1 = 0, indexC2 = 0;
    for (int i = i1; i <= i2; ++i) {
        child1[indexC1++] = parent1[i];
        child2[indexC2++] = parent2[i];

        // the children contain the cities parent[i]
        childOneContains[parent1[i]] = 1;
        childTwoContains[parent2[i]] = 1;
    }

    for (int i = 0; i < genome_size; ++i) {
        // now add only parent2[i] to child1 if it is not already in...
        if (!childOneContains[parent2[i]]) {
            child1[indexC1++] = parent2[i];
            childOneContains[parent2[i]] = 1;
        }

        // same for parent1.
        if (!childTwoContains[parent1[i]]) {
            child2[indexC2++] = parent1[i];
            childTwoContains[parent1[i]] = 1;
        }
    }
}
void MPIMasterSlavePop::init() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    int num_to_alloc = pop_size * genome_size;
    if (rank != 0) num_to_alloc = genome_size * 2;
    next_pop = new int[num_to_alloc];
    pop = new int[num_to_alloc];
    if (rank == 0) {
        fitnesses = new float[pop_size];
        indices = new int[pop_size];
        array_of_requests = new MPI_Request[pop_size * 2];
    }

    //init data randomly
    if (rank == 0) {
        for (int i = 0; i < pop_size; ++i) {
            Random::permutation(pop + i * genome_size, genome_size);
        }
    }
}
void MPIMasterSlavePop::solveProblem(int num_gens) {
    double total = 0.0;
    double totalLoop = 0.0;
    double B = 0.0;
    double C = 0.0, TOTAL = 0.0;
    double TT = 0;
    double MPIB = MPI_Wtime();
    double MPID = 0, MPIC = 0, MPIE = 0, MPIF = 0;
    for (int i = 0; i < num_gens; ++i) {
        // first evaluate all;
        double LOOPSTART = MPI_Wtime();
        float total_myscores = 0.0f;

        if (rank == 0) {
            for (int k = 0; k < pop_size; ++k) {
                int rank_to_send_to = (k % (numranks - 1)) + 1;
                int* to_send = pop + k * genome_size;
                // send TO nodes
                // MPI_Status stat;
                MPI_Isend(to_send, genome_size, MPI_INT, rank_to_send_to, k, MPI_COMM_WORLD, &array_of_requests[k]);
                MPI_Irecv(fitnesses + k, 1, MPI_FLOAT, rank_to_send_to, k, MPI_COMM_WORLD, &array_of_requests[k + pop_size]);
            }
            // now sent all
            double b = MPI_Wtime();
            MPI_Waitall(2 * pop_size, array_of_requests, MPI_STATUSES_IGNORE);
            double e = MPI_Wtime();
            totalLoop += e - b;
        } else {
            for (int k = rank - 1; k < pop_size; k += numranks - 1) {
                // receive data
                MPI_Recv(pop, genome_size, MPI_INT, 0, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // eval
                float score = 1 / evaluateSingle(prob, pop);
                total_myscores += score;
                // float score = 0;
                MPI_Send(&score, 1, MPI_FLOAT, 0, k, MPI_COMM_WORLD);
            }
        }
        float recvbuf;
        double b = MPI_Wtime();
        MPI_Reduce(&total_myscores, &recvbuf, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        double e = MPI_Wtime();
        TOTAL += e - b;
        // now max and second max:
        int max = 0;
        int second_max = 1;
        if (rank == 0) {
            double b = MPI_Wtime();
            for (int i = 2; i < pop_size; i++) {
                // Is it the max?
                if (fitnesses[i] > fitnesses[max]) {
                    // Make the old max the new 2nd max.
                    second_max = max;
                    // This is the new max.
                    max = i;
                }
                // It's not the max, is it the 2nd max?
                else if (fitnesses[i] > fitnesses[second_max]) {
                    second_max = i;
                }
            }
            if (i % 1000 == 0)
                printf("Best = %f\n", 1 / fitnesses[max]);
            double e = MPI_Wtime();
            totalLoop += e - b;
        }
        // send stuff
        MPIC -= LOOPSTART - MPI_Wtime();
        b = MPI_Wtime();
        MPI_Bcast(rank == 0 ? pop + max * genome_size : pop, genome_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(rank == 0 ? pop + second_max * genome_size : pop + genome_size, genome_size, MPI_INT, 0, MPI_COMM_WORLD);
        e = MPI_Wtime();
        B += e - b;
        // now do breeding
        MPID -= LOOPSTART - MPI_Wtime();
        if (rank == 0) {
            double b = MPI_Wtime();
            for (int i = 1; i < pop_size / 2; ++i) {
                indices[i * 2] = Random::random0n(2);
                indices[i * 2 + 1] = Random::random0n(2);
                int rank_to_send_to = ((i - 1) % (numranks - 1)) + 1;
                MPI_Isend(indices + i * 2, 2, MPI_INT, rank_to_send_to, i, MPI_COMM_WORLD, &array_of_requests[i - 1]);

                MPI_Irecv(next_pop + i * 2 * genome_size, 2 * genome_size, MPI_INT, rank_to_send_to, i, MPI_COMM_WORLD, &array_of_requests[i - 2 + pop_size / 2]);
            }
            memcpy(next_pop, pop + max * genome_size, genome_size * sizeof(int));
            memcpy(next_pop + genome_size, pop + second_max * genome_size, genome_size * sizeof(int));
            double e = MPI_Wtime();
            C += e - b;
            b = MPI_Wtime();
            MPI_Waitall(pop_size - 2, array_of_requests, MPI_STATUSES_IGNORE);
            e = MPI_Wtime();
            total += e - b;
            std::swap(next_pop, pop);
        } else {
            for (int i = rank; i < pop_size / 2; i += numranks - 1) {
                int twoindices[2];
                MPI_Recv(twoindices, 2, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // breed these two
                if (twoindices[0] == twoindices[1]) {
                    memcpy(next_pop, pop + twoindices[0] * genome_size, genome_size * sizeof(int));
                    memcpy(next_pop + genome_size, pop + twoindices[1] * genome_size, genome_size * sizeof(int));
                } else
                    salesmanCrossover(pop + twoindices[0] * genome_size, pop + twoindices[1] * genome_size,
                                      next_pop, next_pop + genome_size, genome_size);
                if (Random::random01() < 0.5)
                    mutate_salesman_individual(next_pop, genome_size);
                if (Random::random01() < 0.5)
                    mutate_salesman_individual(next_pop + genome_size, genome_size);
                MPI_Send(next_pop, 2 * genome_size, MPI_INT, 0, i, MPI_COMM_WORLD);
            }
        }
    }
    TT = MPI_Wtime() - MPIB;
    if (!rank) {
        printf("OVERALL = %lf, Total time wait = %lf. Loop Time = %lf.BC = %lf, C = %lf. tota = %lf\n", TT * 1000, total * 1000, totalLoop * 1000, B * 1000, C * 1000, TOTAL * 1000);
        printf("Test = %lf, %lf, %lf.FINAL LOOP = %lf\n", MPIC, MPID, MPIE, MPIF);
    }
}
MPIMasterSlavePop::~MPIMasterSlavePop() {
    if (next_pop)
        delete[] next_pop;
    if (pop)
        delete[] pop;
    if (fitnesses)
        delete[] fitnesses;
    if (array_of_requests)
        delete[] array_of_requests;
    if (indices) delete[] indices;
}
