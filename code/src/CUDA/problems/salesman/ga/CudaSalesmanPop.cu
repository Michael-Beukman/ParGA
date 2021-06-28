#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <unistd.h>

#include "CUDA/problems/salesman/ga/CudaSalesmanPop.h"
#include "CUDA/problems/salesman/CudaSalesmanUtils.h"
#include "CUDA/utils/CudaRandom.h"

void CudaSalesmanPop::evaluate() {
    cuda_evaluate<<<dimGrid, dimBlock, 0>>>(dev_pop, dev_probabilities, population_size, genome_size, device_cities);
}

__global__ void cuda_evaluate(int *dev_pop, float *dev_probabilites, int pop_size, int genome_size, const float *dev_cities) {
    // Assume as many threads as individuals.
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < pop_size) {
        Individual<int> thing_to_eval = getIndividual(i, dev_pop, genome_size);
        // using constant memory for the cities didn't help much.
        float answer = evaluateSingle(thing_to_eval, genome_size, dev_cities);
        dev_probabilites[i] = 1 / answer;
    }
}

void CudaSalesmanPop::init_data_randomly() {
    cuda_init_data_randomly<<<dimGrid, dimBlock, 0>>>(dev_pop, population_size, genome_size, dev_random);
    checkCudaErrors(cudaDeviceSynchronize());
}

/**
 * This should crossover parent1 and parent2 to obtain child1 and child2.
 * We perform crossover by choosing two random indices i1 and i2, and then making the children as follows:
 * 
 * child1: parent1[i1: i2] + parent2[:i1] + parent2[i2:]
 * child2: parent2[i1: i2] + parent1[:i1] + parent1[i2:]
 * while ensuring no duplicates.
 * 
 */
__device__ void crossover(const Individual<int> parent1, const Individual<int> parent2,
                          Individual<int> child1, Individual<int> child2, int genome_size,
                          curandState *dev_random, mybool *tempArray) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    // this is the bit array that is in effect split up into a bunch of ints.
    // For this, for a genome size of 1000, we use 32 ints, as 32 * 32 = 1024 >= 1000
    int number_of_ints_each = ((genome_size - 1) / (sizeof(mybool) * 8) + 1);
    
    // does child1 contain city i
    mybool *childOneContains = tempArray;
    // does child2 contain city j
    mybool *childTwoContains = &tempArray[number_of_ints_each];
    
    // how many bits are in one int
    int S = sizeof(mybool) * 8;

    int i1 = cuda_rand_0n(index, dev_random, genome_size);
    int i2 = cuda_rand_0n(index, dev_random, genome_size);
    if (i1 > i2) {
        // swap
        int temp = i1;
        i1 = i2;
        i2 = temp;
    }
    int indexC1 = 0, indexC2 = 0;
    for (int i = i1; i <= i2; ++i) {
        child1[indexC1++] = parent1[i];
        child2[indexC2++] = parent2[i];
        
        // Very weird way to do the accessing because it is in bit form. We are in effect converting the 1D index
        // parent1[i] into a 2d index, which represents (which int, which bit). 
        int p1 = parent1[i] / S;
        int p1b = parent1[i] % S;
        
        // the children contain the cities parent[i]
        // This is in effect doing the same as saying childOneContains[parent1[i]] = 1, 
        // but the bit ops are necessary 
        childOneContains[p1] |= (1 << p1b);


        // similarly here.
        int p2 = parent2[i] / S;
        int p2b = parent2[i] % S;
        childTwoContains[p2] |= (1 << p2b);
    }

    for (int i = 0; i < genome_size; ++i) {
        int p1 = parent1[i] / S;
        int p1b = parent1[i] % S;

        int p2 = parent2[i] / S;
        int p2b = parent2[i] % S;

        // now add only parent2[i] to child1 if it is not already in...
        // This is checking if childOneContains[parent2[i]] is already 1. If it is, then we don't add in this city,
        // otherwise we do.
        if (!(childOneContains[p2] & (1 << p2b))) {
            child1[indexC1++] = parent2[i];
        }

        // same for parent1.
        if (!(childTwoContains[p1] & (1 << p1b))) {
            child2[indexC2++] = parent1[i];
        }
    }
}
__device__ void mutate(Individual<int> child, float prob, int genome_size, curandState *dev_random, int temp) {
    if (cuda_rand_01(threadIdx.x + blockIdx.x * blockDim.x, dev_random) < prob)
        cuda_tsp_mutate(child, genome_size, dev_random, temp);
}

// Function pointers
__device__ crossover_func<int> device_salesman_crossover = crossover;
__device__ mutation_func<int> device_salesman_mutation = mutate;

void CudaSalesmanPop::init() {
    CudaPopulation<int>::init();
    int num_bytes = sizeof(float) * C * 2;
    checkCudaErrors(cudaMalloc(&device_cities, num_bytes));
    checkCudaErrors(cudaMemcpy(device_cities, problem.positions, num_bytes, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpyFromSymbol(&on_host_mutation, device_salesman_mutation, sizeof(device_salesman_mutation)));
    checkCudaErrors(cudaMemcpyFromSymbol(&on_host_crossover, device_salesman_crossover, sizeof(device_salesman_crossover)));
}