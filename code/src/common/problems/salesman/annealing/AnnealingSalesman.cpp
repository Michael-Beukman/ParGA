#include "common/problems/salesman/annealing/AnnealingSalesman.h"

#include <memory.h>

#include <string>
#include <cmath>
#include "common/problems/salesman/SalesmanUtils.h"

void SimulatedAnnealingSalesman::mutate() {
    // copy current_sol as starting point.
    memcpy(otherSol, current_sol, sizeof(int) * genome_size);
    // and mutate normally.
    mutate_salesman_individual(otherSol, genome_size);
}

void SimulatedAnnealingSalesman::init_data_randomly() {
    // valid solution = random permutation
    Random::permutation(current_sol, genome_size);
    // initialise cost.
    currentCost = costOfOneIndividual(current_sol);
}

float SimulatedAnnealingSalesman::costOfOneIndividual(Individual<int> current_sol) {
    // for SA, we minimise energy, so return distance directly.
    return evaluateSingle(problem, current_sol);
}