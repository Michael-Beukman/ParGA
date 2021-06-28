#ifndef __ROSENBROCK_H__
#define __ROSENBROCK_H__
#include "common/serial/Individual.h"
/**
 * @brief Simple multi-dimensional rosenbrock problem.
 * 
 */
struct Rosenbrock{
    int N;
};

float rosenbrock_evaluate(const Rosenbrock& problem, const float* solution);
bool is_individual_valid(const Individual<float>& solution, const Rosenbrock& problem);
#endif // __ROSENBROCK_H__