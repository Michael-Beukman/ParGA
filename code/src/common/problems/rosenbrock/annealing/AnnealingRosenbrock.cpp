#include "common/problems/rosenbrock/annealing/AnnealingRosenbrock.h"

#include <memory.h>
#include <stdlib.h>

#include <cmath>
#include <string>

#include "common/utils/random/Random.h"

/**
 * @brief Mutate by perturbing two numbers slightly.
 * 
 */
void SimulatedAnnealingRosenbrock::mutate() {
    // We found that only changing 2 numbers performed well.
    memcpy(otherSol, current_sol, sizeof(float) * genome_size);
    int num_to_change = 2;
    for (int i = 0; i < num_to_change; ++i) {
        int i1 = Random::random0n(genome_size);
        otherSol[i1] += (Random::random01() - 0.5) * 0.1;
    }
}
// Inits this individual.
void SimulatedAnnealingRosenbrock::init_data_randomly() {
    Random::rand_floats(current_sol, genome_size, -5, 5);
    currentCost = costOfOneIndividual(current_sol);
}

// The energy for one indiv.
float SimulatedAnnealingRosenbrock::costOfOneIndividual(Individual<float> current_sol) {
    float score = rosenbrock_evaluate(problem, current_sol);
    return score;
}