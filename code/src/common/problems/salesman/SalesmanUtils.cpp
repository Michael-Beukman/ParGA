#include "common/problems/salesman/SalesmanUtils.h"

#include <assert.h>

#include <vector>
#include "common/utils/random/Random.h"

/**
 * @brief Checks if the TSP solution is valid.
 * 
 * @param solution 
 * @param problem 
 * @return true 
 * @return false 
 */
bool is_individual_valid(const Individual<int> solution, const TSP& problem) {
    std::vector<int> counter(problem.C, 0);
    for (int i = 0; i < problem.C; ++i) {
        if (solution[i] < 0 || solution[i] >= problem.C) {
            printf("Term i = %d, not correct (%d)\n", i, solution[i]);
            assert(1 == 0);
        }
        ++counter[solution[i]];
    }
    for (int i = 0; i < problem.C; ++i) {
        if (counter[i] != 1) {
            return false;
        }
    }
    return true;
}
/**
 * @brief Mutates the individual by randomly reversing a range.
 * 
 * @param child 
 * @param genome_size 
 */
void mutate_salesman_individual(Individual<int> child, const int genome_size) {
    int index1 = (Random::random0n(genome_size));
    int index2 = (Random::random0n(genome_size));
    if (index1 == index2)
        return;
    else if (index2 < index1)
        std::swap(index1, index2);

    // Reverse a range, idea was originally from here:
    // http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/TemperAnneal/KirkpatrickAnnealScience1983.pdf
    // And here: S. Lin, B. W. Kernighan An Effective Heuristic Algorithm for the Traveling-Salesman Problem. Operations Research 21 (2) 498-516 https://doi.org/10.1287/opre.21.2.498
    for (int i = index1; i <= index2; ++i) {
        // now reverse
        int j = i - index1;
        if (index2 - j < i) break;
        std::swap(child[i], child[index2 - j]);
    }
}