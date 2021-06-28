#ifndef __CUDAANNEALINGROSENBROCK_H__
#define __CUDAANNEALINGROSENBROCK_H__

#include "CUDA/sa/CudaSimulatedAnnealing.h"
#include "common/problems/rosenbrock/rosenbrock.h"

class CudaAnnealingRosenbrock : public CudaSimulatedAnnealing<float> {
   public:
    float* device_cities = 0;

    // necessary function pointers.
    cuda_get_energy<float> on_host_energy; 
    cuda_mutation_func<float> on_host_mutation;

    const Rosenbrock& problem;
    CudaAnnealingRosenbrock(int _genome_size, const Rosenbrock& _problem, int _block_size=32, int _grid_size=128)
        : CudaSimulatedAnnealing<float>(_genome_size, _block_size, _grid_size),
          problem(_problem) {
            currentTemp = currentTemp * problem.N * problem.N * 10;

          }

    void init_data_randomly() override;
    void init() override;

    virtual cuda_get_energy<float> get_energy_func();
    virtual cuda_mutation_func<float> get_mutation_func();
};

#endif // __CUDAANNEALINGROSENBROCK_H__