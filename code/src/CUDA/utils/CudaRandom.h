#ifndef __CU_RANDOM_H__
#define __CU_RANDOM_H__
#include <curand.h>
#include <curand_kernel.h>

/** Contains some functions that are similar to the Random:: class. This uses the NVIDIA curand library though.
 *  Each function needs to take in the dev_random state.
 * 
 * Uses the curand library, here: https://developer.nvidia.com/curand#:~:text=The%20NVIDIA%20CUDA%20Random%20Number,cores%20available%20in%20NVIDIA%20GPUs.
 * And some other resources: https://stackoverflow.com/questions/15297168/random-generator-cuda
*/

__device__ int cuda_random_choice(float* probabilities, int size, curandState* dev_random, int index);

__device__ void cuda_permutation(int* data, int size, curandState* dev_random, int index);

/**
 * @brief Get the init cuda rand objectInitialises and creates the curandState. Called on the host. 
 * 
 * @param total_number_of_threads Maximum number of threads that will use this random state.
 * @return curandState* 
 */
curandState* get_init_cuda_rand(int total_number_of_threads);

/**
 * @brief Destroys and cleans up the state.
 * 
 * @param dev_random 
 */
void cuda_rand_destroy(curandState* dev_random);

/**
 * @brief Seeds the cuda random state, using index. Could be more elaborate and use the iteration number, but this is simple.
 * 
 * @param index 
 * @param dev_random 
 */
__device__ void cuda_rand_seed(int index, curandState* dev_random);


__device__ float cuda_rand_01(int index, curandState* dev_random);

/**
 * Generates a random int between 0 and n(exclusive); 
 */
__device__ int cuda_rand_0n(int index, curandState* dev_random, int n);
#endif  // __CU_RANDOM_H__