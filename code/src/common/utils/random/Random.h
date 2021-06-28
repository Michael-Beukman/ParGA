#ifndef __RANDOM_H__
#define __RANDOM_H__
#include "common/vendor/random/include/stdfin/random/threefry_engine.hpp"
#define USE_THREEFRY 1
/**
 * @brief A random class that provides some static methods for generating random numbers.
 * It can either use Threefry (from https://github.com/stdfin/random), which is the default. 
 * It can also use the vanilla rand(), which might be useful as a comparison, by doing #define USE_THREEFRY 0
 */
class Random {
#if USE_THREEFRY
// If we have threefry, then create this engine.
   static stdfin::threefry_engine<uint32_t> random_engine;
#endif
   public:
   /**
    * @brief Returns an integer index by choosing randomly with probabilities as specified in probabilities, which has size `size`
    * 
    * @param probabilities 
    * @param size 
    * @return int 
    */
    static int random_choice(float* probabilities, int size);

    /**
     * @brief Creates a permutation from 0 to size -1 and stores it in data[0: size)
     * 
     * @param data 
     * @param size 
     */
    static void permutation(int* data, int size);

    /**
     * @brief Returns a number between 0 and 1
     * 
     * @return float 
     */
    static float random01();

    /**
     * @brief Returns a random number between 0 (inclusive) and n (exclusive)
     * 
     * @param n 
     * @return int 
     */
    static int random0n(int n);

    /**
     * @brief Fills up data with size floats ranging from min to max.
     * 
     * @param data 
     * @param size 
     * @param min 
     * @param max 
     */
    static void rand_floats(float* data, int size, float min, float max);

    /**
     * @brief Seeds the underlying generator.
     * 
     * @param seed 
     */
    static void seed(unsigned int seed);
};
#endif  // __RANDOM_H__