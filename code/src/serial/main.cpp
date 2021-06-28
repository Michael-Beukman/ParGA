#include <assert.h>

#include <chrono>
#include <iostream>

#include "common/experiments/Experiment.h"
#include "common/metrics/metrics.h"
#include "common/problems/rosenbrock/annealing/AnnealingRosenbrock.h"
#include "common/problems/rosenbrock/ga/RosenbrockPop.h"
#include "common/problems/rosenbrock/rosenbrock.h"
#include "common/problems/salesman/SalesmanUtils.h"
#include "common/problems/salesman/TSP.h"
#include "common/problems/salesman/annealing/AnnealingSalesman.h"
#include "common/problems/salesman/ga/SalesmanPop.h"
#include "common/problems/salesman/validation/brute_force.h"
#include "common/utils/random/Random.h"
#include "common/utils/utils.h"
#ifdef DEMO
const bool should_write = false;
#define EXP_VERBOSE 0
#else
const bool should_write = true;
#endif

#ifdef LOG_ALL
const bool write_to_text_file = should_write;
#else
const bool write_to_text_file = false;
#endif
// Where to save results
std::string directory = "serial_test";
// what version, determines specific folder path
const std::string VERSION = "v32_serial_good_exps_0623_test";
// the date.
std::string date = get_date();

// Get an annealing function
template <typename T, typename Problem>
using serial_get_annealing_pointer = std::function<SimulatedAnnealing<T> *(int problem_size, const Problem &prob)>;

// Get a GA function
template <typename T, typename Problem>
using serial_get_population_pointer = std::function<Population<T> *(int problem_size, const Problem &prob, int population_size)>;

// specific functions regarding {GA, SA} x {TSP, Rosen}
Results<int> serial_sa_tsp(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);
Results<int> serial_ga_tsp(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);
Results<float> serial_sa_rosen(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);
Results<float> serial_ga_rosen(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);

// These two actually run the code, but are templated so that the above can call them
template <typename T, typename Problem>
Results<T> serial_do_all_sa(const Problem &problem, serial_get_annealing_pointer<T, Problem> get_annealing_ptr, int problem_size, int seed, int numranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment, const std::string &prob_desc);
template <typename T, typename Problem>
Results<T> serial_do_all_gen_alg(const Problem &problem, serial_get_population_pointer<T, Problem> get_pop_pointer, int problem_size, int seed, int numranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment, const std::string &prob_desc);

/**
 * @brief This runs the serial SA exps using the specified config.
 * 
 * @tparam T 
 * @param config 
 */
template <typename T>
void good_serial_sa_exps(const ExperimentConfigs config) {
    int MAX_ITERS, minSize, maxSize, mult;
    CSVHandler<T> handler;
    std::string problem;
    if (config.is_rosen) {
        problem = "Rosen";
        MAX_ITERS = 1000000;
        minSize = 2;
        maxSize = 64;
        mult = 2;
    } else {
        CSVHandler<int> handler;
        problem = "TSP";
        MAX_ITERS = 1000000;
        minSize = 10;
        maxSize = 1000;
        mult = 10;
    }

    if (config.which_exp == 1) {
        printf("SA, exp %d, problem = %s\n", config.which_exp, problem.c_str());
        directory = "exp1_standard_" + problem;
        for (int num_iters_method = 100; num_iters_method <= MAX_ITERS; num_iters_method *= 10) {
            for (int problem_size = minSize; problem_size <= maxSize; problem_size *= mult) {
                for (int seed = 42; seed < 42 + 3; ++seed) {
                    printf("\rIters %d Prob %d Seed %d                            ", num_iters_method, problem_size, seed);

                    std::string row = "";
                    if (config.is_rosen) {
                        auto results = serial_sa_rosen(problem_size, seed, 1, num_iters_method, -1, 10);
                        row = get_result_row(results, 1, -1, -1, seed, problem, problem_size, results.num_iterations, results.result_filename, "SERIAL_SA");
                    } else {
                        auto results = serial_sa_tsp(problem_size, seed, 1, num_iters_method, -1, 10);
                        row = get_result_row(results, 1, -1, -1, seed, problem, problem_size, results.num_iterations, results.result_filename, "SERIAL_SA");
                    }
                    handler.get_result_row(row);
                }
            }
        }
    }
    printf("\n");
    if (should_write) {
        handler.to_file("proper_exps/" + VERSION + "/serial/sa/" + directory + "_" + problem, "serial");
    }
}

/**
 * @brief This runs the serial GA exps.
 * 
 * @tparam T 
 * @param config 
 */
template <typename T>
void good_serial_ga_exps(const ExperimentConfigs config) {
    int MAX_ITERS, minSize, maxSize, mult;
    CSVHandler<T> handler;
    std::string problem;
    if (config.is_rosen) {
        problem = "Rosen";
        MAX_ITERS = 1000;
        minSize = 2;
        maxSize = 64;
        mult = 2;
    } else {
        CSVHandler<int> handler;
        problem = "TSP";
        MAX_ITERS = 10000;
        minSize = 10;
        maxSize = 1000;
        mult = 10;
    }
    int size_ = 1;
    // which experiment. This can be:
    /*
    1: Standard. Goes from problem size 10 - 1000, iterations from c - d and pop size from e - f. Is basically strong scaling too.
    2: Weak scaling: Constant problem size, iterations range from a - b, and pop size from e - f, but pop_size = pop_size * ranks, 
            so that each rank has a 'constant' amount of pop size.
    */
    if (config.which_exp == 1) {
        printf("GenAlg, exp %d, problem = %s\n", config.which_exp, problem.c_str());
        directory = "ga_exp1_standard_" + problem;
        for (int num_iters_method = 10; num_iters_method <= MAX_ITERS; num_iters_method *= 10) {
            for (int problem_size = minSize; problem_size <= maxSize; problem_size *= mult) {
                for (int population_size = 28 * 2; population_size <= 28 * 4; population_size *= 2) {
                    for (int seed = 42; seed < 42 + 3; ++seed) {
                        printf("\rIters %d Prob %d Seed %d Pop Size %d                          ",
                               num_iters_method, problem_size, seed, population_size);

                        if (config.is_rosen) {
                            auto results = serial_ga_rosen(problem_size, seed, size_, num_iters_method, population_size, 10);

                            handler.get_result_row(get_result_row<float>(results, size_, -1, -1, seed, problem, problem_size, num_iters_method, results.result_filename, "GenAlg", population_size));
                        } else {
                            auto results = serial_ga_tsp(problem_size, seed, size_, num_iters_method, population_size, 10);

                            handler.get_result_row(get_result_row<int>(results, size_, -1, -1, seed, problem, problem_size, num_iters_method, results.result_filename, "GenAlg", population_size));
                        }
                    }
                }
            }
        }
    }
    printf("\n");
    if (should_write) {
        handler.to_file("proper_exps/" + VERSION + "/serial/ga/" + directory + "_" + problem, "serial");
    }
}

/**
 * @brief Runs the demo.
 * 
 * @param argc 
 * @param argv 
 */
void demo(int argc, char **argv) {
    Random::seed(0);
    int size = 1;
    int C = 1000;
    auto problem = createProblem(C, -10, 10);

    auto iters_string = parse_demo_args(argc, argv, size);

    int num_iters_total = std::get<0>(iters_string);
    int num_iters = std::get<1>(iters_string);
    auto str = std::get<2>(iters_string);
    printf("Running TSP with total number of iterations = %d, iterations per rank = %d, scaling = %s. Number of ranks = %d (Serial)\n", num_iters_total, num_iters, str.c_str(), size);

    for (int iter = 0; iter < 3; ++iter) {
        Random::seed(iter * 5000);
        auto ann = SimulatedAnnealingSalesman(C, problem);
        ann.verbose = 0;

        ann.init();
        auto begin = std::chrono::high_resolution_clock::now();
        ann.solveProblem(num_iters);
        auto end = std::chrono::high_resolution_clock::now();
        auto val = ann.get_final_best_solution();
        bool is_valid = is_individual_valid(val.data(), problem);
        double time_in_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        printf("\tTime taken = %lfs. Score at end = %f. Result is %s\n", time_in_ms / 1e3, ann.all_scores.back(), is_valid ? "Valid" : "Invalid");
    }
}

/**
 * @brief Wrapper that calls one of good_serial_sa_exps or good_serial_ga_exps.
 * 
 * @param argc 
 * @param argv 
 */
void serial_run_all_experiments_according_to_args(int argc, char **argv) {
    const ExperimentConfigs config = parse_all_experiment_args(argc, argv);
    if (config.is_sa) {
        if (config.is_rosen)
            good_serial_sa_exps<float>(config);
        else
            good_serial_sa_exps<int>(config);
    } else {
        if (config.is_rosen)
            good_serial_ga_exps<float>(config);
        else
            good_serial_ga_exps<int>(config);
    }
}

int main(int argc, char **argv) {
#ifdef DEMO
    demo(argc, argv);
#else
    serial_run_all_experiments_according_to_args(argc, argv);
#endif
    return 0;
}

// Calls function to run GA + TSP
Results<int> serial_ga_tsp(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Random::seed(seed);
    const TSP prob = createProblem(problem_size, -10, 10);
    auto results = serial_do_all_gen_alg<int, TSP>(
        prob, [](int problem_size, const TSP &prob, int population_size) {
            return new SalesmanPopulation(population_size, prob.C, prob);
        },
        problem_size, seed, ranks, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, prob.description());
    return results;
}

// Calls function to run SA + TSP
Results<int> serial_sa_tsp(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Results<int> res;
    Random::seed(seed);
    const TSP prob = createProblem(problem_size, -10, 10);
    auto results = serial_do_all_sa<int, TSP>(
        prob, [](int problem_size, const TSP &prob) {
            return new SimulatedAnnealingSalesman(prob.C, prob);
        },
        problem_size, seed, ranks, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, prob.description());
    return results;
}

// Rosen ones.
Results<float> serial_sa_rosen(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Results<float> res;
    Random::seed(seed);
    const Rosenbrock prob = Rosenbrock{problem_size};
    auto results = serial_do_all_sa<float, Rosenbrock>(
        prob, [](int problem_size, const Rosenbrock &prob) {
            return new SimulatedAnnealingRosenbrock(problem_size, prob);
        },
        problem_size, seed, ranks, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, "Rosen" + std::to_string(problem_size));
    return results;
}

Results<float> serial_ga_rosen(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Random::seed(seed);
    const Rosenbrock prob = {problem_size};
    auto results = serial_do_all_gen_alg<float, Rosenbrock>(
        prob, [](int problem_size, const Rosenbrock &prob, int population_size) {
            return new RosenbrockPopulation(population_size, problem_size, prob);
        },
        problem_size, seed, ranks, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, "Rosen_" + std::to_string(problem_size));
    return results;
}

/**
 * @brief This actually performs the experiment.
 * 
 * @tparam T                                            Float or Int, type of individual
 * @tparam Problem                                      Rosen or TSP
 * @param problem                                       The problem
 * @param get_annealing_ptr                             The way to create an SA solver.
 * @param problem_size                                  Size of the problem
 * @param seed                                          Seed with which to seed each run.
 * @param numranks                                      The number of ranks in total. For serial and cuda, this should be -1 or 1.
 * @param num_iterations_for_method                     Number of iterations to perform
 * @param pop_size                                      Pop size for GA, SA should make this -1.
 * @param num_iters_for_single_run_experiment           The number of runs to average over. 10 is a good value. Must be greater than 1, as the first run is discarded as a warmup.
 * @param prob_desc                                     A string representing the problem description.
 * @return Results<T>                                   Results representing the score and time taken averaged over (num_iters_for_single_run_experiment-1) runs.
 */
template <typename T, typename Problem>
Results<T> serial_do_all_sa(const Problem &problem, serial_get_annealing_pointer<T, Problem> get_annealing_ptr,
                            int problem_size, int seed, int numranks, int num_iterations_for_method, int pop_size,
                            int num_iters_for_single_run_experiment, const std::string &prob_desc) {
    // lambda function to run.
    auto func = [&](int iter_number) {
        Results<T> res;
        // seed each iteration differently.
        Random::seed(seed * 100 + iter_number * 5000);

        const Problem &prob = problem;
        // timing
        auto begin = std::chrono::high_resolution_clock::now();
        SimulatedAnnealing<T> *pop = get_annealing_ptr(problem_size, prob);
        // initialise and measure how long that took.
        pop->init();
        auto end = std::chrono::high_resolution_clock::now();
        res.time_ms_to_initialise = std::chrono::duration<double, std::milli>(end - begin).count();

        // solve the problem and measure time it takes.
        begin = std::chrono::high_resolution_clock::now();
        pop->solveProblem(num_iterations_for_method);
        end = std::chrono::high_resolution_clock::now();
        res.time_ms_to_perform_operations = std::chrono::duration<double, std::milli>(end - begin).count();

        // some result manging
        double time_in_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        double d = time_in_ms * 1000.0;
        // number of operations, not used.
        long long num_ops = get_number_of_operations(prob, num_iterations_for_method, numranks);
        // save scores
        res.scores = pop->get_all_measured_scores();
        // Save other info
        res.num_iterations = num_iterations_for_method;
        res.num_procs = numranks;
        res.total_num_operations = num_ops;

        // store final soltuion
        res.final_result = pop->get_final_best_solution();
        // assert that the individual is valid.
        bool is_valid = is_individual_valid(res.final_result.data(), prob);
        assert(1 == is_valid);
// Print out depending on mode and defines.
#if EXP_VERBOSE
        printf("Time taken = %lfs for %lld ops, and %'dM gen_pop per second. Score at end = %f. Is valid = %i\n", d / 1e6, num_ops, (int)((num_ops) / d), res.scores.back(), is_valid);
#endif
#ifdef DEMO
        printf("\tTime taken = %lfs. Score at end = %f. Result is %s\n", d / 1e6, res.scores.back(), is_valid ? "Valid" : "Invalid");
#endif
        // delete the pointer we created previously.
        delete pop;
        return res;
    };

    // setup experiment
    Experiment<T> e("Serial, SA.\n" + prob_desc, "seeds_" + std::to_string(seed), func);
    // make directory
    auto pos = "proper_exps/" + VERSION + "/serial/" + directory + "/" + date + "_" + std::to_string(numranks) + "/";
    if (write_to_text_file)
        int _ = system(("mkdir -p " + pos).c_str());
    // run and return results.
    auto results = e.run(pos, num_iters_for_single_run_experiment, write_to_text_file);
    return results;
}
// Very similar to above, just different types.
template <typename T, typename Problem>
Results<T> serial_do_all_gen_alg(const Problem &problem, serial_get_population_pointer<T, Problem> get_pop_pointer, int problem_size, int seed, int numranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment, const std::string &prob_desc) {
    int rank;
    auto func = [&](int iter_number) {
        Results<T> res;
        Random::seed(seed * 100 + iter_number * 5000);

        const Problem &prob = problem;
        auto begin = std::chrono::high_resolution_clock::now();
        Population<T> *pop = get_pop_pointer(problem_size, prob, pop_size);

        pop->init();

        auto end = std::chrono::high_resolution_clock::now();
        res.time_ms_to_initialise = std::chrono::duration<double, std::milli>(end - begin).count();

        begin = std::chrono::high_resolution_clock::now();
        pop->solveProblem(num_iterations_for_method);
        end = std::chrono::high_resolution_clock::now();
        res.time_ms_to_perform_operations = std::chrono::duration<double, std::milli>(end - begin).count();

        double time_in_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        double d = time_in_ms * 1000.0;
        long long num_ops = get_number_of_operations(prob, num_iterations_for_method, numranks);
        res.scores = pop->get_all_measured_scores();
        res.num_iterations = num_iterations_for_method;
        res.num_procs = numranks;
        res.total_num_operations = num_ops;

        res.final_result = pop->get_final_best_solution();
        bool is_valid = is_individual_valid(res.final_result.data(), prob);
        assert(1 == is_valid);
#if EXP_VERBOSE
        if (rank == 0) {
            printf("Time taken = %lfs for %lld ops, and %'dM gen_pop per second. Score at end = %f. Is valid = %i\n", d / 1e6, num_ops, (int)((num_ops) / d), res.scores.back(), is_valid);
        }
#endif
        delete pop;
        return res;
    };
    Experiment<T> e("SERIAL, GA.\n" + prob_desc, "seeds_" + std::to_string(seed), func);
    auto pos = "proper_exps/" + VERSION + "/serial/ga/" + directory + "/" + date + "_serial" + "/";
    if (write_to_text_file)
        int _ = system(("mkdir -p " + pos).c_str());
    auto results = e.run(pos, num_iters_for_single_run_experiment, write_to_text_file);
    return results;
}
