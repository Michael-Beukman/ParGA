#ifndef __MPIANNEALING_H__
#define __MPIANNEALING_H__

#include "common/serial/SimulatedAnnealing.h"


/**
 * @brief MPI Annealing, similar to MPIPopulation. Uses the serial class for most of the shared functionality, 
 *          and communicates every now and then.
 * 
 * @tparam T 
 */
template <typename T>
class MPIAnnealing {
public:
    SimulatedAnnealing<T>* serial_class;
    int myrank, numranks;
    MPIAnnealing(SimulatedAnnealing<T>* _serial_class) : serial_class(_serial_class) {
        serial_class->verbose=0;
    }

    void init();
    void solveProblem(int iteration_count);

    std::vector<float> get_all_measured_scores() {
        return serial_class->get_all_measured_scores();
    }
    std::vector<T> get_final_best_solution(){
        return serial_class->get_final_best_solution();
    }
};

#endif  // __MPIANNEALING_H__