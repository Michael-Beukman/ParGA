#ifndef __SALESMANPOP_H__
#define __SALESMANPOP_H__
#include "common/serial/Individual.h"
#include "common/serial/Population.h"
#include "common/problems/salesman/TSP.h"
class SalesmanPopulation : public Population<int> {
   public:
    const TSP& problem;
    SalesmanPopulation(int pop_size, int _genome_size, const TSP& _problem) : Population(pop_size, _genome_size), problem(_problem) {}

    virtual void crossover(const Individual<int> parent1, const Individual<int> parent2,
                           Individual<int> child1, Individual<int> child2) override;

    virtual void mutate(Individual<int> child, float prob) override;

    virtual float evaluateSingle(const Individual<int> child) override;
    ~SalesmanPopulation() {
    }

    virtual void init_data_randomly() override;
};

#endif  // __SALESMANPOP_H__