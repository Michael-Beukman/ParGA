#include <assert.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

#include <chrono>

#include "CUDA/ga/CudaPopulation.h"
#include "CUDA/problems/salesman/CudaSalesmanUtils.h"

// Performs the whole process.
template <typename T>
void CudaPopulation<T>::solveProblem(int num_gens) {
    for (int i = 0; i < num_gens; ++i) {
        // evaluate all individuals
        evaluate();
        checkCudaErrors(cudaDeviceSynchronize());
        
        // Since we only use the top 2 to breed, don't normalise the probabilities

        // breed next generation
        
        breed();
        checkCudaErrors(cudaDeviceSynchronize());
        // save scores.
        saveScores();
        checkCudaErrors(cudaDeviceSynchronize());
        
        // swap pointers to current and next pops.
        std::swap(dev_pop, dev_next_pop);
    }
}

template <typename T>
void CudaPopulation<T>::init() {
    // malloc on host
    mem_next_pop = (T *)malloc(sizeof(T) * population_size * genome_size);
    mem_pop = (T *)malloc(sizeof(T) * population_size * genome_size);

    // probabilities and population memory on device.
    checkCudaErrors(cudaMalloc(&dev_probabilities, sizeof(float) * population_size));
    checkCudaErrors(cudaMalloc(&dev_mem_pop, sizeof(T) * population_size * genome_size));
    checkCudaErrors(cudaMalloc(&dev_mem_next_pop, sizeof(T) * population_size * genome_size));

    checkCudaErrors(cudaMalloc(&device_total, sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_total_max, sizeof(float)));

    pop = mem_pop;
    next_pop = mem_next_pop;

    dev_pop = dev_mem_pop;
    dev_next_pop = dev_mem_next_pop;

    // initialise cuda_random.
    dev_random = get_init_cuda_rand(population_size);
    // all scores
    checkCudaErrors(cudaMalloc(&dev_global_all_scores, sizeof(float) * NUM_SCORE_POINTS));
    // number of steps so far.
    checkCudaErrors(cudaMalloc(&dev_current_count_of_steps, sizeof(int)));
    int zero = 0;
    checkCudaErrors(cudaMemcpy(dev_current_count_of_steps, &zero, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    init_data_randomly();
    checkCudaErrors(cudaDeviceSynchronize());
}

template <typename T>
CudaPopulation<T>::~CudaPopulation() {
    // Free everything
    cuda_rand_destroy(dev_random);
    free(mem_next_pop);
    free(mem_pop);
    checkCudaErrors(cudaFree(dev_probabilities));
    checkCudaErrors(cudaFree(dev_mem_pop));
    checkCudaErrors(cudaFree(dev_mem_next_pop));
    checkCudaErrors(cudaFree(device_total));
    checkCudaErrors(cudaFree(dev_total_max));
    checkCudaErrors(cudaFree(dev_global_all_scores));
    checkCudaErrors(cudaFree(dev_current_count_of_steps));
}

// Simply calls main_cuda_breed
template <typename T>
void CudaPopulation<T>::breed() {

    // Number of bytes for the bool arrays, as well as for the two top parents.
    int num = (genome_size - 1) / (sizeof(mybool) * 8) + 1;
    const int number_of_bytes = 2 * num * sizeof(mybool) * dimBlock.x + genome_size * sizeof(T) * 2 + population_size * sizeof(int);
    main_cuda_breed<<<dimGridHalf, dimBlockHalf, number_of_bytes>>>(dev_pop, dev_next_pop, dev_probabilities, population_size, genome_size, dev_random,
                                                                    get_crossover_func(),
                                                                    get_mutation_func(),
                                                                    dev_total_max);
}

/**
 * @brief Main function to breed dev_pop into dev_next_pop.
 * 
 */
template <typename T>
__global__ void main_cuda_breed(T *dev_pop, T *dev_next_pop, float *dev_probabilites_normalised, int population_size, int genome_size,
                                curandState *dev_random,
                                crossover_func<T> function_to_do_crossover,
                                mutation_func<T> function_to_do_mutation,
                                float *dev_total_max) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    const int tId = threadIdx.x;
    // This shared memory has the following:
    // 2 bool arrays (each of size genome_size) for each thread in this block, as well as the genomes of the top two parents.
    // Since bools are stored in chars, which wastes 7 bits, we use mybool and bit packing to
    // in effect store a bool array as a collection of integers, with much less wastage.

    // The bool arrays are mainly an optimisation for the crossover of the TSP, which needs a way to keep track of
    // which children contain which cities. Using malloc inside that function is much, much slower than doing it this way.
    // It is a bit less general, as Rosenbrock doesn't need the memory, but in general, I think methods could make use of this.

    // For example, if the genome size is 1000, we need 32 integers to store one boolean array
    //    (each integer has 32 bits, and 32 * 32 = 1024). We do in effect waste those last 24 bits,
    //    but it's more of a headache to use them than not.

    extern __shared__ mybool tempArrays[];
    // This calculates the number of integers we need to store at least `genome_size` bools. Basically ceil(genome_size/(32 bits))
    // We multiply this by two as each thread has access to two arrays.
    int number_of_bits_each = ((genome_size - 1) / (sizeof(mybool) * 8) + 1) * 2;
    // total number of bits for the boolean arrays = blockDim.x * (bits per thread).
    int number_of_bits = number_of_bits_each * blockDim.x;

    // The rest of this array will contain the top two parents from the previous generation, because we only use
    // those for crossover. Again, this results in a massive speedup over using the global memory.

    // copy parents in
    int num_elems_to_copy = genome_size * 2;
    int num_passes = 1 + ((num_elems_to_copy - 1) / blockDim.x);
    T *topTwoParents = (T *)&tempArrays[number_of_bits];

    for (int i = 0; i < num_passes; ++i) {
        int index = tId + i * blockDim.x;
        if (index < num_elems_to_copy) {
            topTwoParents[index] = dev_pop[index];
        }
    }
    // set initial boolean arrays to 0.
    num_passes = 1 + ((number_of_bits - 1) / blockDim.x);
    for (int i = 0; i < num_passes; ++i) {
        int index = tId + i * blockDim.x;
        if (index < number_of_bits) {
            // copy to global mem starting point.
            tempArrays[index] = 0;
        }
    }

    __syncthreads();

    // If we have an invalid thread index, then return
    if (index >= population_size / 2)
        return;

    // This copies the best two individuals into the first two spots for the next generation.
    // doing this reduction over an entire block is faster than only using one thread.

    // First we find the best two individuals, and then copy them over.
    // Note, we find the second best score somewhat fuzzily, and it is not guaranteed to
    // always be exactly the second best score. This is because the main reduction can find the max
    // quite easily, but the second max not so much.
    // We basically take the second max to be one of the intermediate maxes in the reduction. It's not worth it
    // to sort the array first, and it doesn't hamper results that much.

    if (blockIdx.x == 0) {
        // The third part in this shared array is a list of indices, so we can keep track of which genomes are the best.
        int *indices = (int *)&topTwoParents[num_elems_to_copy];

        float *probs = dev_probabilites_normalised;
        // first set up
        num_passes = 1 + ((population_size - 1) / blockDim.x);
        for (int i = 0; i < num_passes; ++i) {
            int index = tId + i * blockDim.x;
            if (index < population_size) {
                // put index in
                indices[index] = index;
            }
        }
        __shared__ int second_max;
        if (index == 0)
            second_max = -1;

        __syncthreads();
        // do a reduction, while keeping the index too. This takes the entire array and
        // puts it into an array of size blockDim.x, which is reduced in the following steps.
        for (int i = 0; i < num_passes; ++i) {
            int index = tId + i * blockDim.x;
            if (index < population_size) {
                if (probs[index] > probs[tId]) {
                    probs[tId] = probs[index];
                    indices[tId] = indices[index];
                }
            }
        }

        // now reduce the above
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            __syncthreads();
            if (tId < stride) {
                if (probs[tId + stride] > probs[tId]) {
                    probs[tId] = probs[tId + stride];
                    indices[tId] = indices[tId + stride];
                }
            }
        }

        __syncthreads();
        // make second_max equal to indices[1]. This sometimes results in max == second_max,
        // but it doesn't affect the final score significantly.
        int max = indices[0];

        if (index == 0 && second_max == -1)
            second_max = indices[1];

        // store into shared.
        __shared__ int all_max;
        if (index == 0) {
            all_max = max;
        }
        __syncthreads();

        // Now simply copy over the parents into the new population.
        auto parent1 = getIndividual<T>(all_max, dev_pop, genome_size);
        auto parent2 = getIndividual<T>(second_max, dev_pop, genome_size);
        auto child1 = getIndividual<T>(0, dev_next_pop, genome_size);
        auto child2 = getIndividual<T>(1, dev_next_pop, genome_size);
        int num_passes = 1 + (genome_size - 1) / blockDim.x;
        for (int i = 0; i < num_passes; ++i) {
            int k = i * blockDim.x;
            if (k + tId < genome_size) {
                child1[k + tId] = parent1[k + tId];
                child2[k + tId] = parent2[k + tId];
            }
        }

        __syncthreads();
    }

    // Now, index 0 (only one thread out of all of them), doesn't breed any two indivs, because
    // the top 2 parents were stored in new_pop[0] and new_pop[1].
    if (index == 0)
        *dev_total_max = dev_probabilites_normalised[0];  // update best score.
    else
        for (int i = index * 2; i < (index * 2) + 1; i += 2) {
            // Perform the actual breeding
            // two parents => two offspring.

            // Found that this form of delayed elitism performed the best, much better than
            // choosing from the entire pop in proportion to their fitness.
            // get the two parents
            int parent1Index = cuda_rand_0n(index, dev_random, 2);
            int parent2Index = cuda_rand_0n(index, dev_random, 2);

            Individual<T> child1 = getIndividual<T>((i), dev_next_pop, genome_size);
            Individual<T> child2 = getIndividual<T>((i + 1), dev_next_pop, genome_size);

            Individual<T> parent1 = &topTwoParents[parent1Index * genome_size];
            Individual<T> parent2 = &topTwoParents[parent2Index * genome_size];

            if (parent1Index == parent2Index) {
                // optimisation, memcpy if parents are the same.
                memcpy(child1, parent1, sizeof(T) * genome_size);
                memcpy(child2, parent2, sizeof(T) * genome_size);
            } else {
                // perform crossover parent1 + parent2 = child1, child2.
                // We also pass in the array of booleans to facilitate faster crossover
                function_to_do_crossover(
                    parent1, parent2,
                    child1, child2,
                    genome_size,
                    dev_random,
                    &tempArrays[threadIdx.x * number_of_bits_each]);
            }

            // Mutate the children.
            // empirically it was found that cuda performed better with a higher mutation chance.
            function_to_do_mutation(child1, 1, genome_size, dev_random, index);
            function_to_do_mutation(child2, 1, genome_size, dev_random, index);
        }
}

// Sums up the dev_probs in one block.
__global__ void get_total(float *dev_probs, int N, float *total) {
    __shared__ float sharedTotal;
    if (threadIdx.x == 0) {
        sharedTotal = 0;
    }
    __syncthreads();

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        atomicAdd(&sharedTotal, dev_probs[index]);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(total, sharedTotal);
    }
}

// normalises all fitness values by total.
__global__ void divideAll(float *dev_probs, int N, float *dev_total) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        dev_probs[index] /= *dev_total;
    }
}

template <typename T>
void CudaPopulation<T>::divide() {
    float my0 = 0;
    // save 0 to device_total.
    checkCudaErrors(cudaDeviceSynchronize());
    // First copy 0 to the total
    checkCudaErrors(cudaMemcpy(device_total, &my0, sizeof(float), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    // Sum up, using multiple blocks
    get_total<<<dimGrid, dimBlock, 0>>>(dev_probabilities, population_size, device_total);
    checkCudaErrors(cudaDeviceSynchronize());

    // divide by total
    divideAll<<<dimGrid, dimBlock, 0>>>(dev_probabilities, population_size, device_total);
    checkCudaErrors(cudaDeviceSynchronize());
}

// Stores the Maximum score into the dev_all_scores array.
__global__ void get_total_and_add(const float *dev_max_prob, const float *dev_total_prob, float *dev_all_scores, int *current_score_count) {
    // updates the intermediate scores.
    float total_denormalised = *dev_max_prob;  // * *dev_total_prob;
    dev_all_scores[*current_score_count] = 1 / total_denormalised;
    ++(*current_score_count);
}

template <typename T>
void CudaPopulation<T>::saveScores() {
    // Only one thread does this.
    get_total_and_add<<<1, 1, 0>>>(dev_total_max, device_total, dev_global_all_scores, dev_current_count_of_steps);
}

template <typename T>
std::vector<float> CudaPopulation<T>::get_all_measured_scores() {
    std::vector<float> scores_vec;
    int current_steps;
    checkCudaErrors(cudaMemcpy(&current_steps, dev_current_count_of_steps, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    float *scores = new float[current_steps];

    checkCudaErrors(cudaMemcpy(scores, dev_global_all_scores, sizeof(float) * current_steps, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    scores_vec.reserve(current_steps);
    for (int i = 0; i < current_steps; ++i) {
        scores_vec.push_back(scores[i]);
    }
    delete[](scores);
    return scores_vec;
}

template <typename T>
std::vector<T> CudaPopulation<T>::get_final_best_solution() {
    // The best agent from previous generation is in dev_pop[0]
    return getVectorFromIndividual(dev_pop, genome_size);
}