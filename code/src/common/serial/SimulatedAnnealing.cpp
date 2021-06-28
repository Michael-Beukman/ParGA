#include "common/serial/SimulatedAnnealing.h"
#include <cmath>
#include "common/utils/random/Random.h"
#include "stdio.h"

template <typename T>
float SimulatedAnnealing<T>::getProbability(float currentScore, float nextScore, float temperature) {
    float dT = nextScore - currentScore;
    float prob = exp(-dT / temperature);
    return prob;
}

template <typename T>
bool SimulatedAnnealing<T>::shouldChooseNewIndividual(float currentScore, float nextScore, float temperature) {
    // if next score is better, then always swap. Otherwise swap with probability given above.
    if (nextScore <= currentScore) return true;
    return getProbability(currentScore, nextScore, temperature) > Random::random01();
}

template <typename T>
void SimulatedAnnealing<T>::solveProblem(int iteration_count) {
    for (int i = 0; i < iteration_count; ++i){
        // create a new individual.
        mutate();
        float new_score = costOfOneIndividual(otherSol);
        // if should swap, then swap
        if (shouldChooseNewIndividual(currentCost, new_score, current_temp)){
            std::swap(current_sol, otherSol);
            currentCost = new_score;
        }
        // update temp linearly.
        current_temp *= coolingFactor;
        // push back when necessary.
        if (score_push_back && iteration_count % 100 == 0)
            all_scores.push_back(currentCost);
    }
}

template class SimulatedAnnealing<float>;
template class SimulatedAnnealing<int>;