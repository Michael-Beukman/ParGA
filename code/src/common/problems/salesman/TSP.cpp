#include "common/problems/salesman/TSP.h"
#include "common/utils/random/Random.h"
#include <assert.h>

/**
 * @brief Calculates the total distance travelled for this solution and this problem.
 * 
 * @param problem 
 * @param solution 
 * @return float 
 */
float evaluateSingle(const TSP& problem, const int* solution){
    float distNow = 0;
    // initial city
    float* currentCity = problem.positions + solution[0] * 2;

// If we should validate during execution, mostly for debugging.
#if VALIDATE
    int* counter = (int*)calloc(problem.C, sizeof(int));
    for (int i = 0; i < problem.C; ++i) {
        if (solution[i] <0 || solution[i] >= problem.C){
            printf("Term i = %d, not correct (%d)\n", i, solution[i]);
            assert (1 == 0);
        }
        ++counter[solution[i]];
    }
    for (int i = 0; i < problem.C; ++i) {
        if (counter[i] != 1) {
            for (int i = 0; i < problem.C; ++i) {
                printf("%d ", solution[i]);
            }
            printf("\n");
            free(counter);
            assert(1 == 0);
        }
    }
    free(counter);
#endif
    for (int i = 1; i < problem.C; ++i) {
        float* newPos = problem.positions + solution[i] * 2;
        float dx = newPos[0] - currentCity[0];
        float dy = newPos[1] - currentCity[1];
        distNow += dx * dx + dy * dy;

        currentCity = newPos;
    }
    return distNow;
}
TSP createProblem(int C, float min, float max) {
    // a position for each of the cities
    float* positions = (float*)malloc(2 * C * sizeof(float));
    for (int i = 0; i < C; ++i) {
        float x = Random::random01() * (max - min) + min;
        float y = Random::random01() * (max - min) + min;
        positions[i * 2] = x;
        positions[i * 2 + 1] = y;
    }

    return {C, positions, min, max};
}