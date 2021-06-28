#include "common/utils/random/Random.h"

#include <algorithm>

#include "stdlib.h"
#if USE_THREEFRY
stdfin::threefry_engine<uint32_t> Random::random_engine;
#endif


int Random::random_choice(float* probabilities, int size) {
#if USE_THREEFRY
    float rando = random_engine() * (1.0f / ((float)std::numeric_limits<uint32_t>::max() + 1));
#else
    float rando = (float)(rand()) / (float)((unsigned)RAND_MAX + 1);
#endif

    for (int i = 0; i < size; ++i) {
        rando -= probabilities[i];
        if (rando <= 0) return i;
    }

    return size - 1;
}

void Random::permutation(int* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = i;
    }

    // now shuffle
    for (int i = size - 1; i >= 0; --i) {
//generate a random number [0, i]
#if USE_THREEFRY
        int j = random_engine() % (i + 1);
#else
        int j = rand() % (i + 1);
#endif
        //swap the last element with element at random index
        std::swap(data[i], data[j]);
    }
}

float Random::random01() {
#if USE_THREEFRY
    return random_engine() * (1.0f / std::numeric_limits<uint32_t>::max());
#else
    return (float)(rand()) / (float)RAND_MAX;
#endif
}
int Random::random0n(int n) {
    return (int)((n)*Random::random01()) % n;
}

void Random::rand_floats(float* data, int size, float min, float max) {
    for (int i = 0; i < size; ++i) {
        data[i] = Random::random01() * (max - min) + min;
    }
}

void Random::seed(unsigned int seed) {
    srand(seed);
    Random::random_engine.seed(seed);
}