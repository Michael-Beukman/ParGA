#ifndef __SIM_ANNEAL_H__
#define __SIM_ANNEAL_H__

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

#include "common/serial/Individual.h"

/**
 * @brief Simulated annealing base class. This is the base for the serial code, but the MPI code uses it too.
 * The main structure is:
 * 
 * - Given a starting, random solution, mutate it slightly.
 * - Change to this solution either if it is better, or with probability exp(-(nextScore - currentScore) / temperature).
 * - Repeat. The temperature gets reduced at every iteration by 0.995.
 * 
 * @tparam T The type of data, i.e. either float or int
 */
template <typename T>
class SimulatedAnnealing {
   public:
    int genome_size;
    // the current solution
    Individual<T> current_sol = nullptr;

    // candidate solution buffer
    Individual<T> otherSol = nullptr;
    
    // annealing specific variables.
    float currentCost;
    float current_temp = 100;
    float coolingFactor = 0.995;
    int numSuccessful = 0;
    int verbose = 0;

    // if true, then pushes scores back to a vector for later analysis.
    bool score_push_back = true;
    
    // All scores to be able to report results later. Only small performance hit.
    std::vector<float> all_scores;

    /**
     * @brief Construct a new Simulated Annealing object
     * 
     * @param _genome_size the size of each individual, the number of variables it requires.
     */
    SimulatedAnnealing(int _genome_size) : genome_size(_genome_size) {}

    /**
     * @brief Should create a random individual as a starting point.
     * 
     */
    virtual void init_data_randomly() = 0;

    /**
     * @brief Allocates memory for the current and candidate sols.
     * 
     */
    virtual void init() {
        current_sol = (T*)malloc(sizeof(T) * genome_size);
        otherSol = (T*)malloc(sizeof(T) * genome_size);
        init_data_randomly();
    }

    /**
     * @brief Calculates the energy from a solution.
     * 
     * @param current_sol 
     * @return float 
     */
    virtual float costOfOneIndividual(Individual<T> current_sol) = 0;

    /**
     * @brief Get the probability to swap
     * 
     * @param currentScore 
     * @param nextScore 
     * @param temperature 
     * @return float 
     */
    float getProbability(float currentScore, float nextScore, float temperature);
    
    /**
     * @brief Returns whether or not the swap should take place.
     * 
     * @param currentScore 
     * @param nextScore 
     * @param temperature 
     * @return true 
     * @return false 
     */
    bool shouldChooseNewIndividual(float currentScore, float nextScore, float temperature);

    /**
     * @brief Solves the problem and performs iteration_count iterations of SA.
     * 
     * @param iteration_count 
     */
    void solveProblem(int iteration_count);

    /**
     * This should mutate current_sol and store it inside otherSol.
     * 
     */
    virtual void mutate() = 0;

    virtual ~SimulatedAnnealing() {
        if (current_sol)
            free(current_sol);
        if (otherSol)
            free(otherSol);
    }

    std::vector<float> get_all_measured_scores() {
        return all_scores;
    }
    std::vector<T> get_final_best_solution() {
        std::vector<T> ans;
        ans.reserve(genome_size);
        for (int i = 0; i < genome_size; ++i) {
            ans.push_back(current_sol[i]);
        }
        return ans;
    }
};

#endif