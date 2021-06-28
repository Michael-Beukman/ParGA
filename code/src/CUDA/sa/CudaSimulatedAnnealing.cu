#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "CUDA/sa/CudaSimulatedAnnealing.h"
#include "CUDA/utils/CudaRandom.h"
#include "assert.h"

// Note, this is set as an int array, but one can easily convert it to a float
// when necessary.
__constant__ int constant_starting_point[MAX_CONSTANT_SIZE];
__constant__ float constant_mem_param[MAX_CONSTANT_SIZE];
// This is the score of the constant_starting_point. Since it was already computed, the threads don't need to compute it again, and they can just read it from here.
__constant__ float constant_mem_starting_score;
__constant__ float coolingFactor = 0.995;

template <typename T>
void CudaSimulatedAnnealing<T>::solveProblem(int iteration_count) {
    int num_things = blockDim.x;
    // 2 individuals (current and candidate), 1 index and 1 float per thread;
    int size_in_bytes;

    if (big_buffer) {
        // if this is non-null, then we don't have enough shared mem to store the individuals.
        // So we only allocate enough for the indices and scores, i.e. one float and one int.
        size_in_bytes = num_things * (sizeof(int) + sizeof(float));
    } else {
        // we have enough shared mem space.
        size_in_bytes = num_things * (2 * sizeof(T) * genome_size + sizeof(int) + sizeof(float));
    }
    for (int i = 0; i < iteration_count; ++i) {
// optional output
#ifdef CUDA_VERBOSE
        if (i % 1000 == 0)
            printf("Running kernel. Size in bytes = %d. Maximum = %d. Big ptr = %p\n", size_in_bytes, max_bytes_shared_mem_available, big_buffer);
#endif
        // run kernel
        iterateSome<T><<<gridDim, blockDim, size_in_bytes>>>(dev_current_sol, dev_current_cost, dev_starting_point, genome_size, dev_random, dev_current_temp, get_mutation_func(), get_energy_func(), device_param, big_buffer);
        checkCudaErrors(cudaDeviceSynchronize());

        // capped at a grid size of 6000, but I think that is plenty.
        int bytes_size_for_reduce = gridDim.x * (sizeof(float) + sizeof(int));

        // find best block results, and save that to dev_starting_point.
        do_reduction_on_global_scores<<<1, gridDim, bytes_size_for_reduce>>>(dev_current_sol, dev_current_cost, gridDim.x, genome_size, dev_current_temp, dev_starting_point,
                                                                             dev_global_all_scores, dev_current_count_of_steps);
        checkCudaErrors(cudaDeviceSynchronize());
        // copy to constant starting point, for fast access.
        checkCudaErrors(cudaMemcpyToSymbol(constant_starting_point, dev_starting_point, sizeof(T) * genome_size, 0, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(constant_mem_starting_score, &dev_current_cost[gridDim.x], sizeof(float), 0, cudaMemcpyDeviceToDevice));
        
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

/**
 * Do we have enough shared mem for everything?
 */
template <typename T>
void CudaSimulatedAnnealing<T>::setup_shared_mem_options() {
    static bool has_warned = false;
    max_bytes_shared_mem_available = cuda_number_of_bytes_shared_mem();
    int bytes_used_for_iteratesome_func = blockDim.x * (2 * sizeof(T) * genome_size + sizeof(int) + sizeof(float));
    // if requested is too much, allocate big_buffer.
    if (bytes_used_for_iteratesome_func > max_bytes_shared_mem_available) {
        // Warn user
        if (!has_warned) {
            printf("\033[1;33m");
            printf("Requested Shared memory size (%dKB) > maximum available (%dKB). Using global memory, which may be slow\n",
                   bytes_used_for_iteratesome_func / 1024, max_bytes_shared_mem_available / 1024);
            printf("\033[0m");
            has_warned = true;
        }

        // each thread (blockDim.x * gridDim.x) has 2 individuals and each indiv is of size genome_size * sizeof(T)
        checkCudaErrors(cudaMalloc(&big_buffer, blockDim.x * gridDim.x * genome_size * 2 * sizeof(T)));
    }
}

template <typename T>
void CudaSimulatedAnnealing<T>::init() {
    // every block will have one score and one individual
    int number_of_individuals = gridDim.x;

    size_t size_sol = genome_size * sizeof(T) * number_of_individuals;
    size_t size_cost = sizeof(float) * number_of_individuals;  // every block will have one score

    // malloc some memory.
    checkCudaErrors(cudaMalloc(&dev_current_sol, size_sol));
    // For the cost, allocate as much as necessary, and one more float for the best score to be stored in and copied to constant mem.
    checkCudaErrors(cudaMalloc(&dev_current_cost, size_cost + sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_current_temp, sizeof(float)));

    checkCudaErrors(cudaMalloc(&dev_starting_point, sizeof(T) * genome_size));

    checkCudaErrors(cudaMalloc(&dev_global_all_scores, sizeof(float) * NUM_SCORE_POINTS));
    checkCudaErrors(cudaMalloc(&dev_current_count_of_steps, sizeof(int)));

    int zero = 0;
    checkCudaErrors(cudaMemcpy(dev_current_count_of_steps, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // random number generator
    dev_random = get_init_cuda_rand(gridDim.x * blockDim.x);
    // check if shared memory is sufficiently large for all our needs. If not, allocate big_buffer for individuals.
    setup_shared_mem_options();

    // Init data
    init_data_randomly();
    checkCudaErrors(cudaDeviceSynchronize());
    // starting point.
    checkCudaErrors(cudaMemcpy(dev_starting_point, dev_current_sol, sizeof(T) * genome_size, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyToSymbol(constant_starting_point, dev_starting_point, sizeof(T) * genome_size, 0, cudaMemcpyDeviceToDevice));
    
    // we save -1 to the constant_mem_starting_score initially, and threads only calculate the score if it is negative. 
    float neg1 = -1;
    checkCudaErrors(cudaMemcpyToSymbol(constant_mem_starting_score, &neg1, sizeof(float)));
    checkCudaErrors(cudaDeviceSynchronize());
}

template <typename T>
CudaSimulatedAnnealing<T>::~CudaSimulatedAnnealing() {
    checkCudaErrors(cudaFree(dev_current_cost));
    checkCudaErrors(cudaFree(dev_current_sol));
    checkCudaErrors(cudaFree(dev_current_temp));
    checkCudaErrors(cudaFree(dev_starting_point));
    checkCudaErrors(cudaFree(dev_global_all_scores));
    checkCudaErrors(cudaFree(dev_current_count_of_steps));
    if (big_buffer) {
        checkCudaErrors(cudaFree(big_buffer));
    }
    cuda_rand_destroy(dev_random);
}

template <typename T>
__global__ void do_reduction_on_global_scores(T* data, float* scores, int number_of_elements, int genome_size, float* dev_temp, T* dev_best_solution,
                                              float* dev_current_scores, int* dev_current_step) {
    //This function runs on a single block and its aim is to determine which score was the best, and
    // copy the corresponding solution from data into starting point.

    extern __shared__ int all_shared_data[];

    int* shared_indices = all_shared_data;
    float* shared_scores = (float*)(&all_shared_data[number_of_elements]);

    int tId = threadIdx.x;

    // save data to the shared
    shared_indices[tId] = tId;
    shared_scores[tId] = scores[tId];

    // Do reduction, while keeping track of index.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tId < stride) {
            if (shared_scores[tId + stride] < shared_scores[tId]) {
                shared_scores[tId] = shared_scores[tId + stride];
                shared_indices[tId] = shared_indices[tId + stride];
            }
        }
    }

    __syncthreads();

    // now put best one inside shared_best_individual_data
    int num_passes = 1 + ((genome_size - 1) / blockDim.x);
    // Pass over individual using all threads to copy.
    for (int i = 0; i < num_passes; ++i) {
        int index = tId + i * blockDim.x;
        if (index < genome_size) {
            // copy to global mem starting point.
            dev_best_solution[index] = data[shared_indices[0] * genome_size + index];
        }
    }

    // Main thread saves current best score to global mem, for reporting purposes, and updates the cooling factor.
    if (tId == 0) {
        *dev_temp = *dev_temp * pow(coolingFactor, NUM_ITERS_IN_BLOCK);
        dev_current_scores[*dev_current_step] = shared_scores[0];
        ++(*dev_current_step);
        scores[number_of_elements] = shared_scores[0];
    }
}

/**
 * Gives the prob to swap from myenergy_for_current to myenergy_for_next given temperature.
 * 
 */
__device__ float cuda_get_prob_to_swap(float myenergy_for_current, float myenergy_for_next, float current_temperature) {
    float dT = myenergy_for_next - myenergy_for_current;
    float prob = exp(-dT / current_temperature);
    return prob;
}

/**
 * Gives the prob to swap from myenergy_for_current to myenergy_for_next given temperature.
 * Returns true if myenergy_for_next <= myenergy_for_current or with SA probability.
 */
__device__ bool should_swap(float myenergy_for_current, float myenergy_for_next, float current_temperature, curandState* dev_random, int id) {
    if (myenergy_for_next <= myenergy_for_current) return true;
    return cuda_get_prob_to_swap(myenergy_for_current, myenergy_for_next, current_temperature) > cuda_rand_01(id, dev_random);
}

/**
 * The meat of this method. each thread performs NUM_ITERS_IN_BLOCK iterations of SA, then the whole block performs a reduction 
 * to determine the best solution, which (along with its score) gets stored in global mem.
 * 
 * There are some optimisations to do with not copying memory when NUM_ITERS_IN_BLOCK = 1, which improved the time taken somewhat,
 * with no effect on the solution
 */
template <typename T>
__global__ void iterateSome(T* data, float* scores, const T* starting_point,
                            int genome_size,
                            curandState* dev_random, float* dev_current_temp,
                            // functions
                            cuda_mutation_func<T> mutate,
                            cuda_get_energy<T> get_energy,
                            const void* param,
                            T* big_buffer_for_individuals) {
    /* 
        Here, we check something
        If the shared memory size = blockDim.x * (2 * sizeof(T) * genome_size + sizeof(int) + sizeof(float)); is greater than the 
        max size (between 48KB and 96KB), then we need to do some stuff differently.
        Then we need shared_mem to only hold the shared_scores and shared_indices (48KB / 8 bytes = 6000, so we're good there)
        If big_buffer_for_individuals is NULL, then all solutions and scores can be inside the shared memory.
        Otherwise, then only the scores and shared indices should be inside the shared mem.
    */
    extern __shared__ int shared_mem[];

    float local_temp = *dev_current_temp;

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int tId = threadIdx.x;

    // individuals to work on
    T* shared_indivs;
    // other buffer for modified indivs.
    T* shared_indivs_buffer;

    // scores of each thread in this block,
    float* shared_scores;
    // indices of each thread, used to determine which thread had the best value.
    int* shared_indices;

    if (big_buffer_for_individuals) {
        // if this is non null, we need to use it to store our intermediate indivs because shared mem was too small.

        shared_indivs = big_buffer_for_individuals + genome_size * blockDim.x * blockIdx.x;
        shared_indivs_buffer = shared_indivs + genome_size * blockDim.x * gridDim.x;

        // these are the only things in shared mem now.
        shared_scores = (float*)&shared_mem[0];
        shared_indices = (int*)&shared_scores[blockDim.x];
    } else {
        // big_buffer_for_individuals == nullptr, so there is definitely enough space in shared mem.
        shared_indivs = (T*)shared_mem;
        shared_indivs_buffer = (T*)(&((T*)shared_mem)[genome_size * blockDim.x]);

        shared_scores = (float*)&shared_mem[genome_size * blockDim.x * 2];
        shared_indices = (int*)&shared_scores[blockDim.x];
    }

    const T* my_indiv = (T*)constant_starting_point;

    // copy into shared mem. Broadcast constant access, which is fast.
    for (int i = 0; i < genome_size; ++i) {
        shared_indivs[tId * genome_size + i] = my_indiv[i];
        shared_indivs_buffer[tId * genome_size + i] = my_indiv[i];
    }
    shared_indices[tId] = tId;

    // don't need this syncthreads as this coming loop is independent.
    Individual<T> my_current_individual = &shared_indivs[threadIdx.x * genome_size];
    Individual<T> my_next_individual = &shared_indivs_buffer[threadIdx.x * genome_size];

    // If we can get the precomputed energy from constant mem, then do so.
    float myenergy_for_current = constant_mem_starting_score;
    // if it is invalid (i.e. on the first iteration, then compute it normally.)
    // We found it's about the same to let each thread compute it than it is to store it in shared mem and only let one thread do it.
    if (myenergy_for_current == -1)
        myenergy_for_current = get_energy(my_current_individual, genome_size, param);

    float og_myenergy_for_current = myenergy_for_current;

    float myenergy_for_next = 0;
    // Loop for NUM_ITERS_IN_BLOCK iterations.
    for (int i = 0; i < NUM_ITERS_IN_BLOCK; ++i) {
        // get candidate solution
        mutate(my_next_individual, genome_size, dev_random, id);
        // what is its energy
        myenergy_for_next = get_energy(my_next_individual, genome_size, param);

        // update local temperature. This is only applicable if we have NUM_ITERS_IN_BLOCK > 1
#if NUM_ITERS_IN_BLOCK > 1
        local_temp *= coolingFactor;
#endif
        // if we should swap, then copy current individual over to next.
        bool do_swap = should_swap(myenergy_for_current, myenergy_for_next, local_temp, dev_random, id);
        if (do_swap) {
            myenergy_for_current = myenergy_for_next;

// Actually relatively important. If we only iterate once, then we don't have to copy, which is faster.
#if NUM_ITERS_IN_BLOCK > 1
            memcpy(my_current_individual, my_next_individual, sizeof(T) * genome_size);
#endif
        } else {
            // if this is the only iter, we don't need to reset next_individual, as it won't be used again.
#if NUM_ITERS_IN_BLOCK > 1
            memcpy(my_next_individual, my_current_individual, sizeof(T) * genome_size);
#endif
        }
    }
    __syncthreads();
    shared_scores[tId] = myenergy_for_current;

    // now do redution in this block.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tId < stride) {
            // reduce both score and index to find the index of the best thread.
            if (shared_scores[tId + stride] < shared_scores[tId]) {
                shared_scores[tId] = shared_scores[tId + stride];
                shared_indices[tId] = shared_indices[tId + stride];
            }
        }
    }
    __syncthreads();
    // now best score and index is in shared_scores[0] and shared_indices[0];
    int my_block_index = genome_size * blockIdx.x;
    // save to global mem.
    if (tId == 0)
        scores[blockIdx.x] = shared_scores[0];
    T* best_shared_one;

    // which buffer should we take? If NUM_ITERS_IN_BLOCK == 1, then we did not copy the potentially better individual to
    // shared_indivs, and it might still be in shared_indivs_buffer.
#if NUM_ITERS_IN_BLOCK > 1
    best_shared_one = &shared_indivs[shared_indices[0] * genome_size];
#else
    // check where the best indiviudal is.
    if (og_myenergy_for_current == shared_scores[0])
        best_shared_one = &shared_indivs[shared_indices[0] * genome_size];
    else
        best_shared_one = &shared_indivs_buffer[shared_indices[0] * genome_size];
#endif

    // now save the best candidate solution to  global memory. All threads take part in this, and it is a coalesced access.
    int num_passes = 1 + ((genome_size - 1) / blockDim.x);
    for (int i = 0; i < num_passes; ++i) {
        int index = tId + i * blockDim.x;
        if (index < genome_size) {
            // copy to global mem starting point.
            data[my_block_index + index] = best_shared_one[index];
        }
    }
}

template <typename T>
std::vector<float> CudaSimulatedAnnealing<T>::get_all_measured_scores() {
    std::vector<float> scores_vec;
    // copy the current step count
    
    int current_steps;
    checkCudaErrors(cudaMemcpy(&current_steps, dev_current_count_of_steps, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    // create a score array.
    float* scores = new float[current_steps];

    // copy from device mem
    checkCudaErrors(cudaMemcpy(scores, dev_global_all_scores, sizeof(float) * current_steps, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    scores_vec.reserve(current_steps);
    // add it to a vector
    for (int i = 0; i < current_steps; ++i) {
        scores_vec.push_back(scores[i]);
    }
    delete[](scores);
    return scores_vec;
}

//////////// Need to specify templates

// Specify the concrete implementations, as it gets used in a different compilation unit.
template __global__ void iterateSome<float>(float* data, float* scores, const float* starting_point,
                                            int genome_size,
                                            curandState* dev_random, float* dev_current_temp,
                                            // functions
                                            cuda_mutation_func<float> mutate,
                                            cuda_get_energy<float> get_energy,
                                            const void* param,
                                            float* big_buffer_for_individuals);

template __global__ void iterateSome<int>(int* data, float* scores, const int* starting_point,
                                          int genome_size,
                                          curandState* dev_random, float* dev_current_temp,
                                          // functions
                                          cuda_mutation_func<int> mutate,
                                          cuda_get_energy<int> get_energy,
                                          const void* param,
                                          int* big_buffer_for_individuals);

// Need to specify for which types it is valid.
template __global__ void do_reduction_on_global_scores<int>(int* data, float* scores, int number_of_elements, int genome_size, float* dev_temp, int* dev_best_solution, float* dev_current_scores, int* dev_current_step);
template __global__ void do_reduction_on_global_scores<float>(float* data, float* scores, int number_of_elements, int genome_size, float* dev_temp, float* dev_best_solution, float* dev_current_scores, int* dev_current_step);
