#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <unistd.h>

#include "CUDA/problems/rosenbrock/ga/CudaRosenbrockPopulation.h"
#include "CUDA/problems/salesman/CudaSalesmanUtils.h"
#include "CUDA/utils/CudaRandom.h"

__global__ void cuda_rosen_evaluate(float *dev_pop, float *dev_probabilites, int pop_size, int genome_size, const void *dev_cities) {
    // Assume as many threads as individuals.
    // make shared mem and better
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < pop_size) {
        Individual<float> thing_to_eval = getIndividual(i, dev_pop, genome_size);
        float answer = cuda_rosenbrock_evaluate(thing_to_eval, genome_size);
        dev_probabilites[i] = 1 / answer;
    }
}
void CudaRosenbrockPop::evaluate() {
    cuda_rosen_evaluate<<<dimGrid, dimBlock, 0>>>(dev_pop, dev_probabilities, population_size, genome_size, device_cities);
}
void CudaRosenbrockPop::init_data_randomly() {
    cuda_init_floats_randomly<<<dimGrid, dimBlock, 0>>>(dev_pop, population_size, population_size, genome_size, dev_random, -5, 5);
    // cuda_init_data_randomly<<<dimGrid, dimBlock, 0>>>(dev_pop, population_size, genome_size, dev_random);
    checkCudaErrors(cudaDeviceSynchronize());
}

/**
     * This should crossover parent1 and parent2 to obtain child1 and child2.
     */
__device__ void crossover(const Individual<float> parent1, const Individual<float> parent2,
                          Individual<float> child1, Individual<float> child2, int genome_size,
                          curandState *dev_random, mybool* tempArray) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int i1 = cuda_rand_0n(index, dev_random, genome_size);
    for (int i = i1; i <= i1; ++i) {
        float beta = cuda_rand_01(index, dev_random);
        child1[i] = beta * parent1[i] + (1 - beta) * parent2[i];
        child2[i] = beta * parent2[i] + (1 - beta) * parent1[i];
    }
    for (int i = 0; i < i1; ++i) {
        child2[i] = parent2[i];
        child1[i] = parent1[i];
    }

    for (int i = i1 + 1; i < genome_size; ++i) {
        child1[i] = parent1[i];
        child2[i] = parent2[i];
    }
}
__device__ void mutate(Individual<float> child, float prob, int genome_size, curandState *dev_random, int index) {
    cuda_rosenbrock_mutate(child, genome_size, dev_random, index);
}

// This is a somewhat hacky way to get function pointers on the device, but specified from the host.
// We first create a device pointer that points to the function.
// This was obtained from: https://forums.developer.nvidia.com/t/how-can-i-use-device-function-pointer-in-cuda/14405/31 
// and https://forums.developer.nvidia.com/t/how-can-i-use-device-function-pointer-in-cuda/14405/32
__device__ crossover_func<float> device_rosenbrock_crossover = crossover;
__device__ mutation_func<float> device_rosenbrock_mutation = mutate;

void CudaRosenbrockPop::init() {
    CudaPopulation<float>::init();
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Then we copy the function from the device symbols above into pointers on the host.
    // This allows us to specify which functions to use on the device, from the host.
    checkCudaErrors(cudaMemcpyFromSymbol(&on_host_mutation, device_rosenbrock_mutation, sizeof(device_rosenbrock_mutation)));
    checkCudaErrors(cudaMemcpyFromSymbol(&on_host_crossover, device_rosenbrock_crossover, sizeof(device_rosenbrock_crossover)));
}