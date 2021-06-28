#ifndef __CUDASALESMANUTILS_H__
#define __CUDASALESMANUTILS_H__
#include "common/serial/Individual.h"
#include <curand_kernel.h>
#include <curand.h>

/**
 * @brief Returns the score for this solution.
 * 
 * @param child 
 * @param C 
 * @param cities 
 * @return __device__ 
 */
__device__ float evaluateSingle(const Individual<int> child, int C, const float *cities);
/**
 * @brief Mutates this individual by swapping ranges.
 
 * Ideas from:
 * http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/TemperAnneal/KirkpatrickAnnealScience1983.pdf
 * And here: S. Lin, B. W. Kernighan An Effective Heuristic Algorithm for the Traveling-Salesman Problem. Operations Research 21 (2) 498-516 https://doi.org/10.1287/opre.21.2.498
 * @param my_next_individual 
 * @param genome_size 
 * @param dev_random 
 * @param index 
 * @return __device__ 
 */
__device__ void cuda_tsp_mutate(Individual<int> my_next_individual, int genome_size, curandState* dev_random, int index);

// Init data randomly, If the size of the data array is the same as the dev_random
__global__ void cuda_init_data_randomly(int *data_to_do, int population_size, int genome_size, curandState *dev_random);

// init salesman data randomly, if the size of the data array is less than the size of dev_random
__global__ void cuda_init_data_randomly_v2(int *data_to_do, int data_size, int random_size, int genome_size, curandState *dev_random);

/** Some rosenbrock utility functions */

__device__ float cuda_rosenbrock_evaluate(const Individual<float> solution, int N);
__device__ void cuda_rosenbrock_mutate(Individual<float> child, int genome_size, curandState* dev_random, int index);
__global__ void cuda_init_floats_randomly(float *data_to_do, int data_size, int random_size, int genome_size, curandState *dev_random, float min, float max);

__device__ bool validate_salesman(int* my_next_individual, int genome_size);
#endif // __CUDASALESMANUTILS_H__