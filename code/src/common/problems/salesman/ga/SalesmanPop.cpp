#include "common/problems/salesman/ga/SalesmanPop.h"

#include "assert.h"
#include "common/problems/salesman/SalesmanUtils.h"
#include "common/utils/utils.h"
#include "common/utils/random/Random.h"

std::string singleRes(int C, int* individual);

float SalesmanPopulation::evaluateSingle(const Individual<int> child) {
    // For a GA, we maximise fitness, so take 1 / distance.
    return 1 / ::evaluateSingle(problem, child);
}

void SalesmanPopulation::init_data_randomly() {
    for (int i = 0; i < population_size; ++i) {
        // A permutation of integers from 0 to genome_size - 1, representing the order in which the cities are visited in.
        Random::permutation(mem_pop + (i * genome_size), genome_size);
    }
}

/**
 * Randomly reverses a range.
 */
void SalesmanPopulation::mutate(Individual<int> child, float prob) {
    // only mutate sometimes.
    if (Random::random01() >= prob) return;
    mutate_salesman_individual(child, genome_size);
}

/**
 * This performs crossover on two individuals. 
 * We perform crossover by choosing two random indices i1 and i2, and then making the children as follows:
 * 
 * child1: parent1[i1: i2] + parent2[:i1] + parent2[i2:]
 * child2: parent2[i1: i2] + parent1[:i1] + parent1[i2:]
 * while ensuring no duplicates.
 * 
 * It then adds the other parent's cities in that order, if they are not already in the child.
 * 
 * Original idea obtained from: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35,
 * and two point crossover from here: [Larranaga et al. 1999] Pedro Larranaga, Cindy M. H. Kuijpers, Roberto H. Murga, Inaki Inza, and Sejla Dizdarevic. Genetic algorithms for the travelling salesman problem: A review of representations and operators. Artificial Intelligence Review, 13(2):129– 170, 1999.
 */
void SalesmanPopulation::crossover(const Individual<int> parent1, const Individual<int> parent2,
                                   Individual<int> child1, Individual<int> child2) {
    int i1 = Random::random0n(genome_size);
    int i2 = Random::random0n(genome_size);

    // v[i] is true if child1 contains city i.

    // Easy and fast way to only add in cities that haven't been added yet.
    // Much better than using O(n^2) to determine if a city exists in a child already.
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
        if (!childOneContains[parent2[i]]){
            child1[indexC1++] = parent2[i];
            childOneContains[parent2[i]] = 1;
        }

        // same for parent1.
        if (!childTwoContains[parent1[i]]){
            child2[indexC2++] = parent1[i];
            childTwoContains[parent1[i]] = 1;
        }
    }
}