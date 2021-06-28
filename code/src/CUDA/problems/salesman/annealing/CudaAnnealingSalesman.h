#ifndef __CUDAANNEALINGSALESMAN_H__
#define __CUDAANNEALINGSALESMAN_H__
#include "CUDA/sa/CudaSimulatedAnnealing.h"
#include "common/problems/salesman/TSP.h"
#include <cmath>

class CudaAnnealingSalesman : public CudaSimulatedAnnealing<int> {
   public:
    float* device_cities = 0;

    cuda_get_energy<int> on_host_energy; 
    cuda_mutation_func<int> on_host_mutation;

    const TSP& problem;
    CudaAnnealingSalesman(int _genome_size, const TSP& _problem, int _block_size=32, int _grid_size=128)
        : CudaSimulatedAnnealing<int>(_genome_size, _block_size, _grid_size),
          problem(_problem) {}
    void init_data_randomly() override;
    void init() override;
    // void solveProblem(int iteration_count) override;


    virtual cuda_get_energy<int> get_energy_func();
    virtual cuda_mutation_func<int> get_mutation_func();
    ~CudaAnnealingSalesman(){
      checkCudaErrors(cudaFree(device_cities));
    }
};

#endif  // __CUDAANNEALINGSALESMAN_H__