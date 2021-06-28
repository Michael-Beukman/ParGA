#ifndef __CUDAUTILS_H__
#define __CUDAUTILS_H__
#include <vector>
#include "common/serial/Individual.h"

/**
 * @brief Swaps to values on the device.
 * 
 * @tparam T 
 * @param val1 
 * @param val2 
 */
template <typename T>
__device__ void cuda_swap(T& val1, T& val2);

template <typename T>
std::vector<T> getVectorFromIndividual(Individual<T> dev_individual, int genome_size);

/**
 * @brief How much shared mem is available.
 * 
 * @return int 
 */
int cuda_number_of_bytes_shared_mem();

#endif  // __CUDAUTILS_H__