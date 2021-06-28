#ifndef __TSP_H__
#define __TSP_H__
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#define VALIDATE 0
/**
 * @brief The travelling salesman problem. Consists of some city positions.
 * 
 */
struct TSP {
    int C;  // number of cities
    float* positions = nullptr;
    float min = -10;
    float max = 10;
    
    std::string description() const{
        std::stringstream ss;
        ss << "TSP Problem. Cities = " << std::to_string(C) << ": ";
        for (int i=0; i<C; ++i){
            float x = positions[i * 2 + 0];
            float y = positions[i * 2 + 1];
            ss << "(" << x << ", " << y << "), ";
        }
        return ss.str();
    }

    ~TSP() {
        if (positions){
            
            free(positions);
            positions =nullptr;
        }
    }
};


/**
 * Returns a floating point number representing the total distance travelled.
 */
float evaluateSingle(const TSP& problem, const int* solution);
/**
 * @brief Randomly creates a TSP problem with the specified number of cities C, and min and max coordinate values.
 * 
 * @param C 
 * @param min 
 * @param max 
 * @return TSP 
 */
TSP createProblem(int C, float min, float max);
#endif // __TSP_H__