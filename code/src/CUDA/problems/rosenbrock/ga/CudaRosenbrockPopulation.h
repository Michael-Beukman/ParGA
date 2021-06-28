
#include "common/serial/Individual.h"
#include "CUDA/ga/CudaPopulation.h"
#include "common/problems/rosenbrock/rosenbrock.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "CUDA/utils/CudaRandom.h"

class CudaRosenbrockPop : public CudaPopulation<float>
{
private:
    int blockSize;


public:
    // assign memory and such things.
protected:
public:
    const Rosenbrock &problem;
    int N;
    float *device_cities = 0;
    int blockSizeMultiple;
    int gridSizeMultiple;

public:
    CudaRosenbrockPop(int pop_size, int _genome_size, const Rosenbrock &_problem) : CudaPopulation<float>(pop_size, _genome_size), problem(_problem)
    {
        // make device problem;
        N = problem.N;
        blockSizeMultiple = 32;
        gridSizeMultiple = ceil(pop_size / (double)blockSizeMultiple / 2.0);
    }

    ~CudaRosenbrockPop() override
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
__device__ void crossover(const Individual<float> parent1, const Individual<float> parent2,
                                  Individual<float> child1, Individual<float> child2, int genome_size,
                                  curandState* dev_random);

/**
     * This should mutate a single individual;
     */
__device__ void mutate(Individual<float> child, float prob, int genome_size, curandState* dev_random);

/**
     * This should return a single fitness value for this individual.
     */
__device__ float evaluateSingle(const Individual<float> child, int C, const void* cities);
