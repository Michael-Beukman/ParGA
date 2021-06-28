#ifndef __CUDASIMULATEDANNEALING_H__
#define __CUDASIMULATEDANNEALING_H__

#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include "CUDA/common_cuda.h"
#include "CUDA/utils/CudaUtils.h"
#include "common/serial/Individual.h"

// number of iterations that each thread does independently. Empirically we found that 1 is much better than using more.
#define NUM_ITERS_IN_BLOCK 1

// The maximum constant size for the cities, and starting point indiv.
#define MAX_CONSTANT_SIZE 15000 / 2

// constant memory parameter, like the cities for TSP.
extern __constant__ float constant_mem_param[MAX_CONSTANT_SIZE];

// mutate single individual
template <typename T>
using cuda_mutation_func = void (*)(Individual<T> individual, int genome_size, curandState* dev_random, int index);

// get energy for one indiv.
template <typename T>
using cuda_get_energy = float (*)(Individual<T> individual, int genome_size, const void* params);

/**
 * @brief This is the main base class for CUDA simulated annealing. It allocates all memory, and runs the code.
 *          This class is abstract, and some functions need to be overwritten by subclasses.
 * 
 * The main idea behind this algorithm in CUDA is the following:
 * 
 * - We have an initial starting point, which is a valid individual.
 * - Each thread performs S iterations of SA on that starting individual, i.e. mutate and (potentially) swap.
 * - Then, each block reduces the results from its threads and stores the best result into global memory.
 * - Then, we perform a global reduction, and the best solution is stored back into the starting point.
 * - The process simply repeats.
 * 
 * @tparam T 
 */
template <typename T>
class CudaSimulatedAnnealing {
   public:
    // general simulated annealing variables, and device versions thereof.
    int genome_size;
    T* dev_current_sol;
    T* dev_starting_point;
    float currentTemp = 100;

    float* dev_current_cost;
    float* dev_current_temp;
    float coolingFactor = 0.995;

    float* dev_global_all_scores;
    int* dev_current_count_of_steps;

    void* device_param;

    // Dimension of the block and grid.
    dim3 blockDim, gridDim;

    // Random state.
    curandState* dev_random;

    int max_bytes_shared_mem_available;

    // if shared mem is too small to hold data.
    T* big_buffer = nullptr;

    // Constructor and destructor
    /**
     * @brief Construct a new Cuda Simulated Annealing object
     * 
     * @param _genome_size  The size of each individual, how many floats or ints does it contain.
     * @param block_size    The requested block size.
     * @param grid_size     The request grid size.
     */
    CudaSimulatedAnnealing(int _genome_size, int block_size, int grid_size) : genome_size(_genome_size), blockDim(block_size), gridDim(grid_size) {}
    ~CudaSimulatedAnnealing();

    /**
     * @brief Determines whether or not we have enough shared memory, and if not, allocates a global buffer to store the candidate solutions.
     * 
     */
    void setup_shared_mem_options();

    /**
     * @brief Returns intermediate scores as a vector.
     * 
     * @return std::vector<float> 
     */
    std::vector<float> get_all_measured_scores();

    /**
     * @brief Returns a single vector representing the best solution found.
     * 
     * @return std::vector<T> 
     */
    std::vector<T> get_final_best_solution() { return getVectorFromIndividual(dev_starting_point, genome_size); }

    /**
     * @brief This performs the main loops, from calling the kernels, and saving scores.
     * 
     * @param iteration_count 
     */
    virtual void solveProblem(int iteration_count);

    /**
     * @brief Initialises and allocates memory. Can be overwritten, but is in practice not.
     * 
     */
    virtual void init();
    

    /**
     * @brief Purely virtual function initialises the data randomly in a problem specific way, for example using permutations for TSP. 
     *  This gets called from init()
     */
    virtual void init_data_randomly() = 0;

    // Get energy and mutation functions.
    virtual cuda_get_energy<T> get_energy_func() = 0;
    virtual cuda_mutation_func<T> get_mutation_func() = 0;
};

// Specify which instantiations are allowed. Not sure how to do this in general, without having the entire code be in the header file.
template class CudaSimulatedAnnealing<int>;
template class CudaSimulatedAnnealing<float>;


//// These are definitions for general free functions that are used in the .cu file.

/**
 * @brief Gets the probability to swap, by using the SA equation.
 * 
 * @param myenergy_for_current 
 * @param myenergy_for_next 
 * @param current_temperature 
 */
__device__ float cuda_get_prob_to_swap(float myenergy_for_current, float myenergy_for_next, float current_temperature);

/**
 * @brief Calculates whether or not we should swap according to the SA rules. Either certainly swap if myenergy_for_next < myenergy_for_current, 
 *              or swap with probability `cuda_get_prob_to_swap`
 * 
 * @param myenergy_for_current 
 * @param myenergy_for_next 
 * @param current_temperature 
 * @param dev_random 
 * @param id 
 */
__device__ bool should_swap(float myenergy_for_current, float myenergy_for_next, float current_temperature, curandState* dev_random, int id);

/**
 * @brief The main simulated annealing function
 * 
* @tparam T                                         float or int depending on the data to be stored.
* @param data                                       An array of gridSize.x individuals corresponding to the best agent from that block.
 * @param scores                                    An array of gridSize.x floats that should contain the scores of each agent in data at the end of this function.
 * @param starting_point                            The starting point solution from which to start the annealing process from. Usually best solution from previous iteration.
 *                                                      Note, we don't use this actually, since we make use of a constant memory starting point instead.
 * @param genome_size                               The size of the genome, i.e. how many numbers does each solution contain
 * @param dev_random                                The random state
 * @param dev_current_temp                          The current temperature
 * @param mutate                                    A function to perform mutation on
 * @param get_energy                                A function to return the energy for a solution
 * @param param                                     A void* that can contain any problem specific information. Actually also not used in favour of constant_mem_param
 * @param big_buffer_for_individuals                If the shared memory is too small to fit all the solutions, 
 *                                                      this pointer should point to global memory that has enough space to accommodate all the solutions.
 *                                                      If there is enough space in shared memory, then this has to be null.
 */
template <typename T>
__global__ void iterateSome(T* data, float* scores, const T* starting_point,
                            int genome_size,
                            curandState* dev_random, float* dev_current_temp,
                            // functions
                            cuda_mutation_func<T> mutate,
                            cuda_get_energy<T> get_energy,
                            const void* param,
                            T* big_buffer_for_individuals);


// Some useful functions for simulated annealing
__device__ float getProbability(float currentScore, float nextScore, float temperature);
__device__ bool shouldChooseNewIndividual(float currentScore, float nextScore, float temperature, curandState* dev_random, int id);

template <typename T>
__device__ float costOfOneIndividual(Individual<T> current_sol);

/**
 * @brief Performs reduction on all the scores in the scores array, which has size `number_of_elements`.
 *  It finds the index with the minimum score, and proceeds to store that genome (stored in data) into the dev_best_solution.
 * 
 *  Index 0 also increments dev_current_step and stores the current best score inside dev_current_scores. It also reduces dev_temp.
 * 
 * Must run on a single block.
 * 
 * @tparam T 
 * @param data 
 * @param scores 
 * @param number_of_elements 
 * @param genome_size 
 * @param dev_temp 
 * @param dev_best_solution 
 * @param dev_current_scores 
 * @param dev_current_step 
 */
template <typename T>
__global__ void do_reduction_on_global_scores(T* data, float* scores, int number_of_elements,
                                              int genome_size, float* dev_temp, T* dev_best_solution,
                                              float* dev_current_scores, int* dev_current_step);


#endif  // __CUDASIMULATEDANNEALING_H__