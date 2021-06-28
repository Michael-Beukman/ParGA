#include "CUDA/problems/rosenbrock/sa/CudaAnnealingRosenbrock.h"
#include "CUDA/utils/CudaRandom.h"
#include "CUDA/problems/salesman/CudaSalesmanUtils.h"
#include "assert.h"

__device__ float cuda_rosenbrock_get_energy(Individual<float> my_current_individual, int genome_size, const void* param) {
    // genome size = N!
    return cuda_rosenbrock_evaluate(my_current_individual, genome_size);
}

__device__ cuda_get_energy<float> device_rosen_energy = cuda_rosenbrock_get_energy;
__device__ cuda_mutation_func<float> device_anneal_rosenbrock_mutation = cuda_rosenbrock_mutate;


cuda_get_energy<float> CudaAnnealingRosenbrock::get_energy_func(){
    return on_host_energy;
}
cuda_mutation_func<float> CudaAnnealingRosenbrock::get_mutation_func(){
    return on_host_mutation;
}


void CudaAnnealingRosenbrock::init_data_randomly() {
    cuda_init_floats_randomly<<<gridDim, blockDim, 0>>>(dev_current_sol, gridDim.x, gridDim.x * blockDim.x, genome_size, dev_random, -5, 5);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(dev_starting_point, dev_current_sol, sizeof(int) * genome_size, cudaMemcpyDeviceToDevice));
}
void CudaAnnealingRosenbrock::init() {
    CudaSimulatedAnnealing<float>::init();

    checkCudaErrors(cudaMemcpy(dev_current_temp, &currentTemp, sizeof(float), cudaMemcpyHostToDevice));

    // copy to function pointers
    checkCudaErrors(cudaMemcpyFromSymbol(&on_host_energy, device_rosen_energy, sizeof(device_rosen_energy)));
    checkCudaErrors(cudaMemcpyFromSymbol(&on_host_mutation, device_anneal_rosenbrock_mutation, sizeof(device_anneal_rosenbrock_mutation)));

    device_param = (void*) NULL;
}
