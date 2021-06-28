#include "CUDA/utils/CudaUtils.h"
#include "helper_cuda.h"

template <typename T>
__device__ void cuda_swap(T& val1, T& val2) {
    T temp = val1;
    val1 = val2;
    val2 = temp;
}
template __device__ void cuda_swap<int>(int&, int&);

template <typename T>
std::vector<T> getVectorFromIndividual(Individual<T> dev_individual, int genome_size) {
    T* dst = new T[genome_size];
    checkCudaErrors(cudaMemcpy(dst, dev_individual, sizeof(T) * genome_size, cudaMemcpyDeviceToHost));
    std::vector<T> vec_indiv;
    vec_indiv.reserve(genome_size);
    for (int i = 0; i < genome_size; ++i){
        vec_indiv.push_back(dst[i]);
    }
    delete[] dst;
    return vec_indiv;
}

template std::vector<int> getVectorFromIndividual(Individual<int> dev_individual, int genome_size);
template std::vector<float> getVectorFromIndividual(Individual<float> dev_individual, int genome_size);


int cuda_number_of_bytes_shared_mem(){
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    return p.sharedMemPerBlock;
}