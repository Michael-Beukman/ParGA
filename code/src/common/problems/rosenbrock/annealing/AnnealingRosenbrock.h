#ifndef __ANNEALINGROSENBROCK_H__
#define __ANNEALINGROSENBROCK_H__

#include "common/serial/SimulatedAnnealing.h"
#include "common/serial/Individual.h"
#include "common/problems/rosenbrock/rosenbrock.h"
#define ROOT2 1.4142135624f
/**
 * @brief Some specific rosenbrock annealing code, like specifics of mutation.
 * 
 */
class SimulatedAnnealingRosenbrock: public SimulatedAnnealing<float> {
public:
    const Rosenbrock& problem;
    SimulatedAnnealingRosenbrock(int _genome_size, const Rosenbrock& _problem): 
        SimulatedAnnealing<float>(_genome_size), problem(_problem){
        // Temperature depdendent on the problem size too.
        current_temp = current_temp * problem.N * problem.N * 10; 
        coolingFactor = 0.995;
    }
    
    virtual void mutate();
    virtual void init_data_randomly();
    virtual float costOfOneIndividual(Individual<float> current_sol);
};

#endif // __ANNEALINGROSENBROCK_H__