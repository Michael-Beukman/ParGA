#include "common/serial/Population.h"

#include "common/utils/utils.h"
#include "common/utils/random/Random.h"

template <typename T>
/**
 * @brief Goes over all individuals, and breeds them.
 * The best two agents will remain unchanged and the rest will be bred to generate the next generation.
 * 
 * @param startIndex 
 * @param endIndex 
 * @param shouldCopyFirstTwo If this is true, then copies the best two agents into either startIndex & startIndex + 1, 
 *                              or  whereToPutBestIndivs and whereToPutBestIndivs + 1.
 * @param whereToPutBestIndivs 
 */
void Population<T>::breed(int startIndex, int endIndex, bool shouldCopyFirstTwo, int whereToPutBestIndivs) {
    if (whereToPutBestIndivs == -1) {
        whereToPutBestIndivs = startIndex;
    }
    int max = 0;
    int second_max = 1;
    if (shouldCopyFirstTwo) {
        // Find max and second max.
        for (int i = 2; i < population_size; i++) {
            // Is it the max?
            if (probabilities[i] > probabilities[max]) {
                // Make the old max the new 2nd max.
                second_max = max;
                // This is the new max.
                max = i;
            }
            // It's not the max, is it the 2nd max?
            else if (probabilities[i] > probabilities[second_max]) {
                second_max = i;
            }
        }
        
        // copy the memory to next pop
        auto bestParent = getIndividual(max, pop);
        auto secondBestParent = getIndividual(second_max, pop);

        memcpy(getIndividual(whereToPutBestIndivs, next_pop), bestParent, sizeof(T) * genome_size);
        memcpy(getIndividual(whereToPutBestIndivs + 1, next_pop), secondBestParent, sizeof(T) * genome_size);
    }
    for (int i = startIndex + shouldCopyFirstTwo * 2; i < endIndex; i += 2) {
        // two parents => two offspring.
        
        // This does quite well. Actually better and faster than choosing an index randomly in proportion to the fitness.
        // So we perform elitism and choose only the top two parents to breed
        int parent1Index = Random::random0n(2) == 0 ? max : second_max;
        int parent2Index = Random::random0n(2) == 0 ? max : second_max;

        Individual<T> child1 = getIndividual((i), next_pop);
        Individual<T> child2 = getIndividual((i + 1), next_pop);
        
        // small optimisation, if parent1 == parent2, don't perform expensive crossover, just copy memory.
        if (parent1Index == parent2Index) {
            memcpy(child1, getIndividual(parent1Index, pop), genome_size * sizeof(T));
            memcpy(child2, getIndividual(parent1Index, pop), genome_size * sizeof(T));
        } else {
            // crossover the two indivs.
            crossover(
                getIndividual(parent1Index, pop),
                getIndividual(parent2Index, pop),
                child1,
                child2);
        }
        // mutate. Empirically it was found that a high mutation probability was beneficial.
        mutate(child1, 1);
        mutate(child2, 1);
    }
}

template <typename T>
void Population<T>::solveProblem(int num_gens) {
    for (int i = 0; i < num_gens; ++i) {
        // evaluate
        float maxRes = evaluate();
#ifdef POPULATION_VERBOSE
        printf("At gen %d, max result = %lf\n", i, 1 / maxRes);
#endif
        // push back if necessary
        if (score_push_back)
            all_scores.push_back(1 / maxRes);
        // breed.
        breed(0, population_size, true);
        // swap pointers to pop and next_pop, so that next_pop is now our current one.
        std::swap(pop, next_pop);
    }
}

template <typename T>
Population<T>::~Population() {
    free(mem_next_pop);
    free(mem_pop);
    free(probabilities);
}
// Specify which instantiations are possible.
template class Population<float>;
template class Population<int>;