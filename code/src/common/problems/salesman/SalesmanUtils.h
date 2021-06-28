#ifndef __SALESMANUTILS_H__
#define __SALESMANUTILS_H__
#include "common/serial/Individual.h"
#include "common/utils/random/Random.h"
#include <algorithm>
#include "common/problems/salesman/TSP.h"


bool is_individual_valid(const Individual<int> solution, const TSP& problem);

void mutate_salesman_individual(Individual<int> child, const int genome_size);
#endif // __SALESMANUTILS_H__