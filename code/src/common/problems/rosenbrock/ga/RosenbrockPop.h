#ifndef __ROSENBROCKPOP_H__
#define __ROSENBROCKPOP_H__


#include "common/serial/Individual.h"
#include "common/serial/Population.h"
#include "common/problems/rosenbrock/rosenbrock.h"

/**
 * @brief Rosenbrock population.
 * 
 */
class RosenbrockPopulation : public Population<float> {
   public:
    const Rosenbrock& problem;
    RosenbrockPopulation(int pop_size, int _genome_size, const Rosenbrock& _problem) : Population(pop_size, _genome_size), problem(_problem) {}

    virtual void crossover(const Individual<float> parent1, const Individual<float> parent2,
                           Individual<float> child1, Individual<float> child2) override;

    virtual void mutate(Individual<float> child, float prob) override;

    virtual float evaluateSingle(const Individual<float> child) override;
    virtual void init_data_randomly() override;
    ~RosenbrockPopulation() {}
};

#endif // __ROSENBROCKPOP_H__