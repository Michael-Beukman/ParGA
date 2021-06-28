#include "CUDA/utils/CudaRandom.h"
#include "helper_cuda.h"

__device__ int cuda_random_choice(float *probabilities, int size, curandState *dev_random, int index) {
    // could potentially optimise this using a cumulative array and binary search.
    float rando = curand_uniform(&dev_random[index]);
    for (int i = 0; i < size; ++i) {
        rando -= probabilities[i];
        if (rando <= 0)
            return i;
    }
    return size - 1;
}

__device__ void cuda_permutation(int *data, int size, curandState *dev_random, int index) {
    for (int i = 0; i < size; ++i) {
        data[i] = i;
    }

    // now shuffle
    for (int i = size - 1; i >= 0; --i) {
        //generate a random number [0, n-1]
        int j = (int)((size) * curand_uniform(&dev_random[index])) % size;

        //swap the last element with element at random index
        int temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

curandState *get_init_cuda_rand(int total_number_of_threads) {
    curandState *dev_random;
    checkCudaErrors(cudaMalloc((void **)&dev_random, total_number_of_threads * sizeof(curandState)));

    return dev_random;
}

__device__ void cuda_rand_seed(int index, curandState *dev_random) {
    curand_init(index, index, 0, &dev_random[index]);
}

void cuda_rand_destroy(curandState* dev_random){
    checkCudaErrors(cudaFree(dev_random));
}

__device__ float cuda_rand_01(int index, curandState* dev_random){
    return curand_uniform(&dev_random[index]);
}

__device__ int cuda_rand_0n(int index, curandState* dev_random, int n){
 return (int)((n) * curand_uniform(&dev_random[index])) % n;   
}