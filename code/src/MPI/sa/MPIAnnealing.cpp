#include "MPI/sa/MPIAnnealing.h"
#include "common/problems/salesman/annealing/AnnealingSalesman.h"
#include <mpi.h>
struct FloatInt {
    float distance;
    int myrank;
};

template <typename T>
void MPIAnnealing<T>::init() {
    serial_class->init();
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
}

template <typename T>
void MPIAnnealing<T>::solveProblem(int iteration_count) {
    // when do we exchange?
    int when_to_exchange = 100;
    // How many times will we exchange.
    int num_blocks = iteration_count / when_to_exchange;
    
    // initial score, to ensure we don't have an empty vector.
    serial_class->all_scores.push_back(10000000);
    for (int i = 0; i < num_blocks; ++i) {
        // each rank runs SA for 100 iterations
        serial_class->solveProblem(when_to_exchange);

        // now we need to check who has the best score, and then broadcast that to every rank.
        // https://www.open-mpi.org/doc/v4.0/man3/MPI_Reduce.3.php
        FloatInt thing_to_send{serial_class->currentCost, myrank};
        FloatInt thing_to_receive;
        
        // MINLOC gives us the score and index of the rank with the lowest energy.
        MPI_Allreduce(&thing_to_send, &thing_to_receive, 1, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);

        // send the new data to all the ranks as a starting point.
        MPI_Bcast(serial_class->current_sol, serial_class->genome_size, MPI_INT, thing_to_receive.myrank, MPI_COMM_WORLD);

        // we know the current cost of this, so simply set it.        
        serial_class->currentCost = thing_to_receive.distance;
        // save results.
        if (!myrank) {
            serial_class->all_scores.push_back(thing_to_receive.distance);
        }
#ifdef MPI_VERBOSE
        if (!myrank) {
            printf("At iter %d, Rank %d had min value of %f, and global min is %f from rank %d\n", (i + 1) * when_to_exchange, myrank, thing_to_send.distance, thing_to_receive.distance, thing_to_receive.myrank);
        }
#endif
    }
}

template class MPIAnnealing<int>;
template class MPIAnnealing<float>;