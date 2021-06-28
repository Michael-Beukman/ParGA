#include <stdio.h>

#include "CUDA/problems/salesman/CudaSalesmanUtils.h"
#include "CUDA/utils/CudaRandom.h"
#include "CUDA/utils/CudaUtils.h"
// Evaluate the TSP solution.
__device__ float evaluateSingle(const Individual<int> child, int C, const float *cities) {
    float distNow = 0;
    // initial city
    const float *currentCity = cities + child[0] * 2;
    for (int i = 1; i < C; ++i) {
        const float *newPos = cities + child[i] * 2;
        float dx = newPos[0] - currentCity[0];
        float dy = newPos[1] - currentCity[1];
        distNow += dx * dx + dy * dy;

        currentCity = newPos;
    }
    return distNow;
}

/**
 * This function does the following
 * Choose two random integers i, j, and
 * subarray my_next_individual[i : j] will become flliped
 * 
 */
__device__ void cuda_tsp_mutate(Individual<int> my_next_individual, int genome_size, curandState *dev_random, int index) {
    int index1 = cuda_rand_0n(index, dev_random, genome_size);
    int index2 = cuda_rand_0n(index, dev_random, genome_size);
    if (index1 == index2) return;
    if (index2 < index1)
        cuda_swap(index1, index2);
    // std::swap(otherSol[index1], otherSol[index2]); return;
    for (int i = index1; i <= index2; ++i) {
        // now reverse
        int j = i - index1;
        if (index2 - j <= i) break;
        cuda_swap(my_next_individual[i], my_next_individual[index2 - j]);
    }
}

__global__ void cuda_init_data_randomly(int *data_to_do, int population_size, int genome_size, curandState *dev_random) {
    // If the size of the data array is the same as the dev_random
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // first init random state
    if (i < population_size) {
        cuda_rand_seed(i, dev_random);
        Individual<int> indiv = data_to_do + i * genome_size;
        cuda_permutation(indiv, genome_size, dev_random, i);
    }
}

__global__ void cuda_init_data_randomly_v2(int *data_to_do, int data_size, int random_size, int genome_size, curandState *dev_random) {
    // If the size of the data array is less than the size of dev_random
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // first init random state
    if (i < random_size) {
        cuda_rand_seed(i, dev_random);
    }
    if (i < data_size) {
        Individual<int> indiv = data_to_do + i * genome_size;
        cuda_permutation(indiv, genome_size, dev_random, i);
    }
}

__device__ float cuda_rosenbrock_evaluate(const Individual<float> solution, int N) {
    float total = 0.0f;
    for (int i = 0; i < N - 1; ++i) {
        float xi = solution[i], xiplus = solution[i + 1];
        total += 100.0f * pow(xiplus - xi * xi, 2) + pow(1 - xi, 2);
    }
    return total;
}
__device__ void cuda_rosenbrock_mutate(Individual<float> child, int genome_size, curandState *dev_random, int index) {
    for (int i = 0; i < 2; ++i) {
        int i1 = cuda_rand_0n(index, dev_random, genome_size);
        child[i1] += (cuda_rand_01(index, dev_random) - 0.5) * 0.1;
    }
}

__global__ void cuda_init_floats_randomly(float *data_to_do, int data_size, int random_size, int genome_size, curandState *dev_random, float min, float max) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // first init random state
    if (i < random_size) {
        cuda_rand_seed(i, dev_random);
    }
    if (i < data_size) {
        Individual<float> indiv = data_to_do + i * genome_size;
        for (int j=0; j<genome_size; ++j){
            indiv[j] = cuda_rand_01(i, dev_random) * (max - min) + min;
        }
    }
}
__device__ bool validate_salesman(int* my_next_individual, int genome_size) {
    int* counts = (int*)malloc(genome_size * sizeof(int));
    for (int i = 0; i < genome_size; ++i) counts[i] = 0;

    for (int i = 0; i < genome_size; ++i) counts[my_next_individual[i]]++;

    bool isBad = false;
    for (int i = 0; i < genome_size; ++i)
        if (counts[i] != 1) {
            printf("Invalid result: %d occurs %d times\n", i, counts[i]);
            isBad = true;
            break;
        }

    free(counts);
    return isBad;
}
