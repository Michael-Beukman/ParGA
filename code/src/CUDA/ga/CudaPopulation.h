#ifndef __CUDA_POPULATION_H__
#define __CUDA_POPULATION_H__
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>
#include <random>
#include "CUDA/common_cuda.h"
#include "CUDA/utils/CudaRandom.h"
#include "CUDA/utils/CudaUtils.h"
#include "common/serial/Individual.h"
#include "helper_cuda.h"
#include "stdlib.h"
#include "string.h"

// Way to pack bits, 32 bools can be stored inside a single uint32_t.
typedef uint32_t mybool;

// A function to crossover parent1 and parent2 into child1 and child2
template <typename T>
using crossover_func = void (*)(const Individual<T> parent1, const Individual<T> parent2,
                                Individual<T> child1, Individual<T> child2, int genome_size,
                                curandState *dev_random, mybool *tempArray);

// A function to mutate child with probability prob.
template <typename T>
using mutation_func = void (*)(Individual<T> child, float prob, int genome_size, curandState *dev_random, int index);

/**
 * @brief Main, super class for CUDA population (i.e. CUDA genetic algorithm). This performs the main work, but it needs subclasses to overwrite some problem specific functions .
 * 
 * The main structure is as follows, first initialise the entire population to random, but valid, individuals.
 * Then, for each generation, we do the following.
 * 
 * 
 * - Evaluate all individuals
 * - Then, block 0 finds the best 2 individuals, and stores them into the next pop at position 0, 1.
 * - And in the meantime, the other blocks (and block 0) breeds parents in positions 0, 1 in the current pop
 * - into new children, which get mutated.
 * - This gets repeated for a number of generations.
 * 
 * @tparam T float or int, depending on the datatype.
 */
template <typename T>
class CudaPopulation {
   protected:
   public:
    
    /**
    * These are the population memory pointers. Current pop and next pop.
    * They are not actually used, since everything is performed on the device, including the initilisation.
    */
    T *mem_pop = NULL;
    T *mem_next_pop = NULL;
    T *pop, *next_pop;

    /* These are similar, just that they are the device data pointers */
    T *dev_mem_pop = NULL;
    T *dev_mem_next_pop = NULL;
    T *dev_pop, *dev_next_pop;

    /* The probabilities to choose a specific individual (on device) */
    float *dev_probabilities = NULL;

    /* Total probabilities on device */
    float *device_total = NULL;

    int population_size;
    int genome_size;

    // These two are simply dimensions for the grid and block. Block dim is 32 & dimgrid = pop size / 32.
    dim3 dimGrid, dimBlock;

    // These are the dimensions if we only use pop_size / 2 threads instead of pop_size.
    // This is for example crossover generates 2 children from 2 parents.
    dim3 dimGridHalf, dimBlockHalf;

    // This is a random state, which can be used to generate random numbers on the gpu.
    curandState *dev_random;

    crossover_func<T> on_host_crossover;
    mutation_func<T> on_host_mutation;

    // keep track of scores.
    float *dev_global_all_scores;
    int *dev_current_count_of_steps;

    // max of all probabilities.
    float *dev_total_max;

   public:
    /**
    * @brief Construct a new Cuda Population object. It uses a block size of 32, so grid size = pop_size / 32.
    * 
    * @param pop_size How big should the population be. Must be at least 64, and a power of two is much easier to deal with.
    * @param _genome_size How many numbers does each individual need.
    */
    CudaPopulation(int pop_size, int _genome_size) : population_size(pop_size), genome_size(_genome_size) {
        // we create two different sizes, for evaluation, one thread per pop
        const int N = 32;
        const int div = (pop_size / 32);
        if(!(div > 0 && div % 2 == 0)) {
            printf("CUDA population size must be at least 64.\n");
            exit(1);
        }
        dimBlock = dim3(N, 1, 1);
        dimGrid = dim3(ceil(pop_size / (double)dimBlock.x), 1, 1);

        // and for breeding, one thread for every two population.
        dimBlockHalf = dim3(N, 1, 1);
        dimGridHalf = ceil(pop_size / (double)dimBlockHalf.x / 2.0);
    }
    /**
     * @brief Destroy the Cuda Population object. Cleans up all of the memory.
     * 
     */
    virtual ~CudaPopulation();
    /**
     * @brief Initialises all of the memory, and calls `init_data_randomly`
     * 
     */
    virtual void init();

    /**
     * @brief Performs the genetic algorithm for num_gens generations.
     * 
     * @param num_gens 
     */
    virtual void solveProblem(int num_gens);
    
    /**
     * @brief Sums up the fitnesses of all individuals, and divides the entire fitness array by that total.
     *          From experimental evidence, we found that using the entire population to breed performed worse,
     *          so we simply take the top two. Thus, we actually don't use this, but it is here for completeness.
     * 
     */
    void divide();
    /**
     * @brief Breeds the current generation into the next generation. Stores the best parents into dev_next_pop[0] and dev_next_pop[genome_size]
     * 
     */
    virtual void breed();

    /**
     * @brief This should evaluate the entire population and store the (un-normalised) scores in dev_probabilities
     * 
     */
    virtual void evaluate() = 0;
    
    /**
     * @brief Saves the best score to an array for reporting later on.
     * 
     */
    void saveScores();


    /**
     * @brief Should initialise the data randomly, called only once, by init(). Should also seed dev_random.
     * 
     */
    virtual void init_data_randomly() = 0;

    /**
     * @brief Return a function pointer that can perform crossover
     * 
     * @return crossover_func<T> 
     */
    crossover_func<T> get_crossover_func() { return on_host_crossover; }
    
    /**
     * 
     * @brief Return a function pointer that can perform mutation
     * 
     * @return mutation_func<T> 
     */
    mutation_func<T> get_mutation_func() { return on_host_mutation; }

    // Copy measured scores from device to host and returns them in a vector.
    std::vector<float> get_all_measured_scores();
    // Returns a vector of the final solution that has the best score.
    std::vector<T> get_final_best_solution();
};

template class CudaPopulation<int>;
template class CudaPopulation<float>;

/**
 * @brief Utility function to get an inidivdual inside a block of memory.
 * 
 * @tparam T 
 * @param index 
 * @param mem_of_first_indiv 
 * @param genome_size 
 * @return Individual<T> Returns a pointer to the start of the individual.
 */
template <typename T>
__device__ Individual<T> getIndividual(int index, T *mem_of_first_indiv, int genome_size) {
    return mem_of_first_indiv + index * genome_size;
}

/**
 * @brief This performs the main breeding, using parents fom dev_pop with associated fitnesses dev_probabilites_normalised, and breeds them into
 *          dev_next_pop. Uses function_to_do_crossover and function_to_do_mutation to perform the crossover and mutation respectively.
 * 
 * @tparam T 
 * @param dev_pop 
 * @param dev_next_pop 
 * @param dev_probabilites_normalised 
 * @param population_size 
 * @param genome_size 
 * @param dev_random 
 * @param function_to_do_crossover 
 * @param function_to_do_mutation 
 */
template <typename T>
__global__ void main_cuda_breed(T *dev_pop, T *dev_next_pop, float *dev_probabilites_normalised, int population_size, int genome_size,
                                curandState *dev_random, crossover_func<T> function_to_do_crossover, mutation_func<T> function_to_do_mutation);

/**
 * @brief Sums up array dev_probs (of size N) into total. Total must be set to 0 initially.
 * 
 * @param dev_probs 
 * @param N 
 * @param total 
 */
__global__ void get_total(float *dev_probs, int N, float *total);

/**
 * @brief Divides every element in dev_probs (which has length N) by dev_total.
 * 
 * @param dev_probs 
 * @param N 
 * @param dev_total 
 * @return __global__ 
 */
__global__ void divideAll(float *dev_probs, int N, float *dev_total);

#endif  // __CUDA_POPULATION_H__
