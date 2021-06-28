#ifndef __POPULATION_H__
#define __POPULATION_H__
#include <random>
#include <vector>

#include "common/serial/Individual.h"
#include "stdlib.h"
#include "string.h"

template <typename T>
/**
 * @brief This class is a base class for each problem-specific population.
 * Contains code to allocate and free memory, as well as an interface to follow for subclasses.
 * The structure of this genetic algorithm is 
 * - Initially create a random population.
 * Then, for each generation
 * - Evaluate the whole pop.
 * - Find the top 2 individuals, and store them in the first two positions of next_pop.
 * - Then, until the next population is full up, breed those two parents, and mutate the children.
 * - Swap current_pop and next_pop, and repeat.
 * 
 */
class Population {
   public:
   // Memory for current and next populations
    T* mem_pop = NULL;
    T* mem_next_pop = NULL;

    // These are pointers to the memory, which can easily be swapped. I think it it possible to not use these at all,
    // and just use the memory pointers directly.
    T *pop, *next_pop;
    
    // the probabilities/fitnesses for each individual.
    float* probabilities = NULL;

    int population_size;
    int genome_size;

    std::vector<float> all_scores;

    // if this is true, then we push back results every 100 iterations.
    bool score_push_back = true;

   public:
    /**
    * @brief Construct a new Population object
    * 
    * @param pop_size  How many individuals should be created
    * @param _genome_size  How many numbers each individual should store.
    */
    Population(int pop_size, int _genome_size) : population_size(pop_size), genome_size(_genome_size) {}

    /**
     * @brief Initialises memory.
     */
    virtual void init() {
        mem_next_pop = (T*)malloc(sizeof(T) * population_size * genome_size);
        mem_pop = (T*)malloc(sizeof(T) * population_size * genome_size);
        probabilities = (float*)malloc(sizeof(float) * population_size);

        pop = mem_pop;
        next_pop = mem_next_pop;
        init_data_randomly();
    }

    /**
     * @brief Populates the existing memory with a random collection of individuals.
     * 
     */
    virtual void init_data_randomly() = 0;

    /**
     * @brief This should crossover parent1 and parent2 to obtain child1 and child2.
     */
    virtual void crossover(const Individual<T> parent1, const Individual<T> parent2,
                           Individual<T> child1, Individual<T> child2) = 0;

    /**
     * @brief This should mutate a single individual.
     */
    virtual void mutate(Individual<T> child, float prob) = 0;

    /**
     * @brief This should return a single fitness value for this individual.
     */
    virtual float evaluateSingle(const Individual<T> child) = 0;

    /**
     * @brief Goes through all scores, divides them by the total and returns the maximum score encountered.
     * 
     * @return float 
     */
    virtual float evaluate() {
        float total = 0;
        float maxVal = 0;
        for (int i = 0; i < population_size; ++i) {
            probabilities[i] = evaluateSingle(getIndividual(i, pop));
            total += probabilities[i];
            maxVal = std::max(maxVal, probabilities[i]);
        }
        for (int i = 0; i < population_size; ++i) {
            probabilities[i] /= total;
        }

        return maxVal;
    }

    /**
     * @brief This should actually breed and create the next generation.
     * 
     * @param startIndex The starting index from which to breed
     * @param endIndex The ending index (exclusive) to which to breed. We only write to the next_pop array in positions [startIndex:endIndex]
     * @param shouldCopyFirstTwo If this is true, then we copy the best two individuals to the next population.
     * @param whereToPutBestIndivs This controls where we place the best individuals. -1 indicates startIndex and startIndex + 1.
     */
    virtual void breed(int startIndex, int endIndex, bool shouldCopyFirstTwo = false, int whereToPutBestIndivs = -1);

    /**
     * @brief Destroy the Population object. Cleans up memory.
     * 
     */
    virtual ~Population();

    /**
     * @brief This performs num_gens generations of the GA.
     * 
     * @param num_gens 
     */
    virtual void solveProblem(int num_gens);

    /**
     * @brief Utility to extract individuals from the memory.
     * 
     * @param index 
     * @param mem_of_first_indiv 
     * @return Individual<T> 
     */
    Individual<T> getIndividual(int index, T* mem_of_first_indiv) {
        return mem_of_first_indiv + index * genome_size;
    }

    /**
     * @brief Returns a vector representing the scores obtained per generation.
     * 
     * @return std::vector<float> 
     */
    std::vector<float> get_all_measured_scores() {
        return all_scores;
    }
    
    /**
     * @brief Get the final best solution as a vector.
     * 
     * @return std::vector<T> 
     */
    std::vector<T> get_final_best_solution() {
        std::vector<T> ans;
        ans.reserve(genome_size);
        for (int i = 0; i < genome_size; ++i) {
            ans.push_back(pop[i]);
        }
        return ans;
    }
};
#endif  // __POPULATION_H__