#include "common/problems/rosenbrock/rosenbrock.h"

#include <cmath>

/**
 * @brief Simply evaluates the function and returns the function value at point x = solution.
 * 
 * @param problem 
 * @param solution 
 * @return float 
 */
float rosenbrock_evaluate(const Rosenbrock& problem, const float* solution) {
    float total = 0.0f;
    for (int i = 0; i < problem.N - 1; ++i) {
        float xi = solution[i], xiplus = solution[i + 1];
        total += 100.0f * pow(xiplus - xi * xi, 2) + pow(1 - xi, 2);
    }
    return total;
}
/**
 * @brief The individual for rosenbrock is always valid, as there isn't really any constraints 
 *          (except the -30 <= xi <= 30, which we ignore for simplicity)
 * 
 * @param solution 
 * @param problem 
 * @return true 
 * @return false 
 */
bool is_individual_valid(const Individual<float>& solution, const Rosenbrock& problem){
    return true;
}