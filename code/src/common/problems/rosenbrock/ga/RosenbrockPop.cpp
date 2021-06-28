#include "common/problems/rosenbrock/ga/RosenbrockPop.h"

#include "common/utils/random/Random.h"

/**
 * @brief Simple rosenbrock crossover, from here: Carr, J. (2014). An Introduction to Genetic Algorithms.
 * 
 * @param parent1 
 * @param parent2 
 * @param child1 
 * @param child2 
 */
void RosenbrockPopulation::crossover(const Individual<float> parent1, const Individual<float> parent2,
                                     Individual<float> child1, Individual<float> child2) {
    int i1 = Random::random0n(genome_size);
    for (int i = i1; i <=i1; ++i) {
        float beta = Random::random01();

        child1[i] = beta * parent1[i] + (1 - beta) * parent2[i];
        child2[i] = beta * parent2[i] + (1 - beta) * parent1[i];
    }
    // return;
    for (int i = 0; i < i1; ++i) {
        child2[i] = parent2[i];
        child1[i] = parent1[i];
    }

    for (int i = i1 + 1; i < genome_size; ++i) {
        child1[i] = parent1[i];
        child2[i] = parent2[i];
    }
}

void RosenbrockPopulation::mutate(Individual<float> child, float prob) {
    // randomly perturbs some values.
    for (int i = 0; i < genome_size; ++i) {
        if (Random::random01() < prob) {
            child[i] += (Random::random01() - 0.5);
        }
    }
}

float RosenbrockPopulation::evaluateSingle(const Individual<float> child) {
    // maximise fitness => minimise function value.
    return 1 / rosenbrock_evaluate(problem, child);
}

void RosenbrockPopulation::init_data_randomly() {
    // random floating point numbers as initial solution.
    for (int i = 0; i < population_size; ++i) {
        Random::rand_floats(mem_pop + (i * genome_size), genome_size, -5, 5);
    }
}