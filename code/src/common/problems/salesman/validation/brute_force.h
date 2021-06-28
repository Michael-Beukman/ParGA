#ifndef __BRUTE_FORCE_H__
#define __BRUTE_FORCE_H__
#include "common/problems/salesman/TSP.h"
#include <vector>
#include <algorithm>

/**
 * @brief Calculates the absolute fastest route for this problem by brute force. 
 * This is only feasible for very small problem sizes.
 * 
 * @param tsp 
 * @return float 
 */
float bestValue(TSP& tsp){
    std::vector<int> ans(tsp.C);
    for (int i=0; i< ans.size(); ++i){
        ans[i] = i;
    }
    float bestScore = 1e8;
    do {
        bestScore = std::min(evaluateSingle(tsp, ans.data()), bestScore);
    } while (std::next_permutation(begin(ans), end(ans)));

    return bestScore;
}

#endif // __BRUTE_FORCE_H__