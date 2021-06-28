#ifndef __SIM_ANNEAL_SALESMAN_H__
#define __SIM_ANNEAL_SALESMAN_H__
#include "common/serial/SimulatedAnnealing.h"
#include "common/serial/Individual.h"
#include "common/problems/salesman/TSP.h"
#include "stdio.h"
#define ROOT2 1.4142135624f

/**
 * @brief Simulated annealing for the TSP problem.
 * 
 */
class SimulatedAnnealingSalesman: public SimulatedAnnealing<int> {
public:
    const TSP& problem;
    SimulatedAnnealingSalesman(int _genome_size, const TSP& _problem): SimulatedAnnealing<int>(_genome_size), problem(_problem){
        // set the current temperature to be proportional to the problem size, as well as the range of positions of the cities.
        current_temp = current_temp * (problem.C) * (ROOT2 * (problem.max - problem.min));
    }
    virtual void mutate();
    virtual void init_data_randomly();
    virtual float costOfOneIndividual(Individual<int> current_sol);
};

#endif