#include "CUDA/problems/salesman/CudaSalesmanUtils.h"
#include "CUDA/problems/salesman/annealing/CudaAnnealingSalesman.h"
#include "CUDA/utils/CudaRandom.h"
#include "assert.h"
__device__ float cuda_salesman_get_energy(Individual<int> my_current_individual, int genome_size, const float* cities) {
    return evaluateSingle(my_current_individual, genome_size, cities);
}

__device__ float cuda_salesman_get_energy(Individual<int> my_current_individual, int genome_size, const void* cities) {
    // using constant memory here, but experiments showed that it's not much faster. Very slight increase.
    // Maybe because dev_cities was already cached.
    return evaluateSingle(my_current_individual, genome_size, (float*)constant_mem_param);
}

__device__ cuda_get_energy<int> device_salesman_energy = cuda_salesman_get_energy;
__device__ cuda_mutation_func<int> device_anneal_salesman_mutation = cuda_tsp_mutate;


cuda_get_energy<int> CudaAnnealingSalesman::get_energy_func(){
    return on_host_energy;
}
cuda_mutation_func<int> CudaAnnealingSalesman::get_mutation_func(){
    return on_host_mutation;
}

// Validates data.
__global__ void cuda_test_data_is_random(int* data, int pop_size, int genome_size){
    validate_salesman(data + threadIdx.x * genome_size, genome_size);
}
void CudaAnnealingSalesman::init_data_randomly() {
    cuda_init_data_randomly_v2<<<gridDim, blockDim, 0>>>(dev_current_sol, gridDim.x, gridDim.x * blockDim.x, genome_size, dev_random);
    checkCudaErrors(cudaDeviceSynchronize());
}
void CudaAnnealingSalesman::init() {
    CudaSimulatedAnnealing<int>::init();
    int num_bytes = sizeof(float) * problem.C * 2;
    checkCudaErrors(cudaMalloc(&device_cities, num_bytes));
    checkCudaErrors(cudaMemcpy(device_cities, problem.positions, num_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(constant_mem_param, problem.positions, num_bytes));
    checkCudaErrors(cudaMemcpy(dev_current_temp, &currentTemp, sizeof(float), cudaMemcpyHostToDevice));

    // setup function pointers.
    checkCudaErrors(cudaMemcpyFromSymbol(&on_host_energy, device_salesman_energy, sizeof(device_salesman_energy)));
    checkCudaErrors(cudaMemcpyFromSymbol(&on_host_mutation, device_anneal_salesman_mutation, sizeof(device_anneal_salesman_mutation)));

    device_param = (void*) device_cities;
}
