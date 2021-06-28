
#include "common/serial/Individual.h"
#include "CUDA/ga/CudaPopulation.h"
#include "common/problems/salesman/TSP.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "CUDA/utils/CudaRandom.h"
__global__ void cuda_breed(int *dev_pop, int *dev_next_pop, float *dev_probabilites_normalised, int population_size, int genome_size,
                            curandState* dev_random);

__global__ void cuda_evaluate(int *dev_pop, float *dev_probabilites, int pop_size, int genome_size, const float *dev_cities);

class CudaSalesmanPop : public CudaPopulation<int>
{
private:
    int blockSize;


public:
    // assign memory and such things.
protected:
public:
    const TSP &problem;
    int C;
    float *device_cities = 0;
    int blockSizeMultiple;
    int gridSizeMultiple;

public:
    CudaSalesmanPop(int pop_size, int _genome_size, const TSP &_problem) : CudaPopulation<int>(pop_size, _genome_size), problem(_problem)
    {
        // make device problem;
        C = problem.C;
        blockSizeMultiple = 32;
        gridSizeMultiple = ceil(pop_size / (double)blockSizeMultiple / 2.0);
    }

    ~CudaSalesmanPop() override
    {
        checkCudaErrors(cudaFree(device_cities));
    }
    
    void init();
    virtual void init_data_randomly();

    virtual void evaluate();
};

/**
     * This should crossover parent1 and parent2 to obtain child1 and child2.
     */
__device__ void crossover(const Individual<int> parent1, const Individual<int> parent2,
                                  Individual<int> child1, Individual<int> child2, int genome_size,
                                  curandState* dev_random);

/**
     * This should mutate a single individual;
     */
__device__ void mutate(Individual<int> child, float prob, int genome_size, curandState* dev_random);

/**
     * This should return a single fitness value for this individual.
     */
__device__ float evaluateSingle(const Individual<int> child, int C, const float* cities);
