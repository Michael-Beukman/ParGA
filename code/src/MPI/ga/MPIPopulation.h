#ifndef __MPIPOPULATION_H__
#define __MPIPOPULATION_H__

#include "common/serial/Population.h"

/**
 * @brief A general MPI population class that implements a genetic algorithm in parallel.
 * It uses a serial population object to perform all the tasks, and only does the MPI communication.
 * This leads to a clean, reusable architecture.
 * 
 * @tparam T 
 */
template <typename T>
class MPIPopulation {
    Population<T>* serial_class;
    const int root = 0;
    int myrank, num_ranks, pop_per_rank, start, end;
    float local_total = 0.0f;
    // How often information is exchanged. One could definitely experiment with this, but we didn't.
    int copy_interval = 10;

   public:
    MPIPopulation(Population<T>* _serial_class) : serial_class(_serial_class) {}
    
    void init();
    void solveProblem(int num_gens);
    float evaluate();

    std::vector<float> get_all_measured_scores() {
        return serial_class->get_all_measured_scores();
    }
    std::vector<T> get_final_best_solution() {
        return serial_class->get_final_best_solution();
    }
};
#endif  // __MPIPOPULATION_H__