#include <mpi.h>
#include <stdio.h>

#include <chrono>


#include <tuple>
#include <type_traits>

#include "MPI/ga/MPIPopulation.h"
#include "MPI/ga/master_slave/MPIMasterSlavePop.h"
#include "MPI/problems/salesman/MPISalesmanPop.h"
#include "MPI/sa/MPIAnnealing.h"
#include "assert.h"
#include "common/experiments/Experiment.h"
#include "common/metrics/metrics.h"
#include "common/problems/rosenbrock/annealing/AnnealingRosenbrock.h"
#include "common/problems/rosenbrock/ga/RosenbrockPop.h"
#include "common/problems/salesman/ga/SalesmanPop.h"
#include "common/problems/salesman/SalesmanUtils.h"
#include "common/problems/salesman/annealing/AnnealingSalesman.h"
#include "common/problems/salesman/validation/brute_force.h"
#include "common/utils/utils.h"
#define DO_SERIAL 0
#ifdef DEMO
#define EXP_VERBOSE 0
const bool should_write = false;
#else
#define EXP_VERBOSE 0
const bool should_write = true;
#endif

#ifdef LOG_ALL
const bool write_to_text_file = should_write;
#else
const bool write_to_text_file = false;
#endif

std::string directory = "mpi_larger_problem_size";
const std::string VERSION = "v38_mpi_good_exps_0627_test_corrected";
std::string date = get_date();

SimulatedAnnealingSalesman *serial = NULL;
SimulatedAnnealingRosenbrock *serial_rosen = NULL;
SalesmanPopulation *serial_salesman_pop = NULL;
RosenbrockPopulation *serial_salesman_pop_rosen = NULL;

template <typename T, typename Problem>
using mpi_get_annealing_pointer = std::function<MPIAnnealing<T> *(int problem_size, const Problem &prob, int num_ranks)>;

template <typename T, typename Problem>
using mpi_get_population_pointer = std::function<MPIPopulation<T> *(int problem_size, const Problem &prob, int num_ranks, int population_size)>;

Results<int> mpi_sa_tsp(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);
Results<int> mpi_ga_tsp(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);
Results<float> mpi_sa_rosen(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);
Results<float> mpi_ga_rosen(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);

template <typename T, typename Problem>
Results<T> mpi_do_all_sa(const Problem &problem, mpi_get_annealing_pointer<T, Problem> get_annealing_ptr, int problem_size, int seed, int numranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment, const std::string &prob_desc);
template <typename T, typename Problem>
Results<T> mpi_do_all_gen_alg(const Problem &problem, mpi_get_population_pointer<T, Problem> get_pop_pointer, int problem_size, int seed, int numranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment, const std::string &prob_desc);
void mpi_run_all_experiments_according_to_args(int argc, char **argv);
template <typename T>
void good_mpi_ga_exps(const ExperimentConfigs config);
template <typename T>
void good_mpi_sa_exps(const ExperimentConfigs config);
void master_slave_ga();

void demo(int argc, char **argv) {
    // parse args
    int seed = 0;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto iters_string = parse_demo_args(argc, argv, size);

    int C = 1000;
    int num_iters_total = std::get<0>(iters_string);
    int num_iters = std::get<1>(iters_string);
    auto str = std::get<2>(iters_string);
    if (!rank)
        printf("Running TSP with total number of iterations = %d, iterations per rank = %d, scaling = %s. Number of ranks = %d\n", num_iters_total, num_iters, str.c_str(), size);
    auto results = mpi_sa_tsp(C, seed, size, num_iters, -1, 3);
    if (!rank)
        printf("\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
#ifdef DEMO
    demo(argc, argv);
#else
    mpi_run_all_experiments_according_to_args(argc, argv);
#endif
    MPI_Finalize();
    return 0;
}

template <typename T>
void good_mpi_sa_exps(const ExperimentConfigs config) {
    int MAX_ITERS, minSize, maxSize, mult;
    int rank_, size_;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    if (config.which_exp == 1) {
        printf("SA, exp %d, problem = %s\n", config.which_exp, problem.c_str());
        directory = "exp1_standard_" + problem;
        for (int num_iters_method = 100; num_iters_method <= MAX_ITERS; num_iters_method *= 10) {
            for (int problem_size = minSize; problem_size <= maxSize; problem_size *= mult) {
                for (int seed = 42; seed < 42 + 3; ++seed) {
                    printf("\rIters %d Prob %d Seed %d                            ", num_iters_method, problem_size, seed);

                    std::string row = "";
                    if (config.is_rosen) {
                        auto results = mpi_sa_rosen(problem_size, seed, size_, num_iters_method, -1, 10);
                        if (!rank_)
                            row = get_result_row(results, size_, -1, -1, seed, problem, problem_size, results.num_iterations, results.result_filename, "MPI_SA");
                    } else {
                        auto results = mpi_sa_tsp(problem_size, seed, size_, num_iters_method, -1, 10);
                        if (!rank_)
                            row = get_result_row(results, size_, -1, -1, seed, problem, problem_size, results.num_iterations, results.result_filename, "MPI_SA");
                    }
                    if (!rank_)
                        handler.get_result_row(row);
                }
            }
        }
    } else if (config.which_exp == 2) {
        printf("Experiment 2 is actually the same as exp 1. Do that rather");
        assert(1 == 0);
    } else if (config.which_exp == 3) {
        printf("SA, exp %d, problem = %s\n", config.which_exp, problem.c_str());
        // Strong scaling. Here we keep the workload constant (e.g. 1k iterations total) and scale up nodes.
        directory = "exp3_strong_scaling";
        int problem_size = 1000;
        for (int num_iters = 100 * 14; num_iters <= MAX_ITERS / 10 * 14; num_iters *= 10) {
            for (int seed = 42; seed < 42 + 3; ++seed) {
                printf("\rIters %d Prob %d Seed %d                            ", num_iters, problem_size, seed);
                // for num_iters = 100 * 16 and ranks = 8, we do 200 iterations. But, each rank in effect does each iteration
                // independently, so this totals to 1600.
                int num_iters_to_do = num_iters / size_;

                std::string row = "";
                if (config.is_rosen) {
                    auto results = mpi_sa_rosen(problem_size, seed, size_, num_iters_to_do, -1, 10);
                    if (!rank_)
                        row = get_result_row(results, size_, -1, -1, seed, problem, problem_size, results.num_iterations, results.result_filename, "MPI_SA");
                } else {
                    auto results = mpi_sa_tsp(problem_size, seed, size_, num_iters_to_do, -1, 10);
                    if (!rank_)
                        row = get_result_row(results, size_, -1, -1, seed, problem, problem_size, results.num_iterations, results.result_filename, "MPI_SA");
                }
                if (!rank_)
                    handler.get_result_row(row);
            }
        }
    } else if (config.which_exp == 4) {
        printf("SA, exp %d, problem = %s\n", config.which_exp, problem.c_str());
        // Strong scaling. Here we keep the workload constant (e.g. 1k iterations total) and scale up nodes.
        directory = "exp4_strong_scaling_corrected";
        for (int problem_size = minSize; problem_size <= maxSize; problem_size *= mult) {
            for (int num_iters = 100 * 14; num_iters <= MAX_ITERS / 10 * 14; num_iters *= 10) {
                for (int seed = 42; seed < 42 + 3; ++seed) {
                    printf("Iters %d Prob %d Seed %d\n", num_iters, problem_size, seed);
                    // for num_iters = 100 * 16 and ranks = 8, we do 200 iterations. But, each rank in effect does each iteration
                    // independently, so this totals to 1600. This experiment does a correction as well, to account for the low frequency of communication.
                    for (double correction: {1.0, 1.5, 2.0, 2.5, 3.0, 4.0}){
                        int num_iters_to_do = (int)(num_iters / size_ * correction);
                        std::string row = "";
                        if (config.is_rosen) {
                            auto results = mpi_sa_rosen(problem_size, seed, size_, num_iters_to_do, -1, 10);
                            if (!rank_)
                                row = get_result_row(results, size_, -1, -1, seed, problem, problem_size, results.num_iterations, results.result_filename, "MPI_SA");
                        } else {
                            auto results = mpi_sa_tsp(problem_size, seed, size_, num_iters_to_do, -1, 10);
                            if (!rank_)
                                row = get_result_row(results, size_, -1, -1, seed, problem, problem_size, results.num_iterations, results.result_filename, "MPI_SA");
                        }
                        if (!rank_)
                            handler.get_result_row(row);
                    }
                }
            }
        }
    }
    printf("\n");
    if (should_write && rank_ == 0) {
        handler.to_file("proper_exps/" + VERSION + "/mpi/sa/" + directory + "_" + problem, "ranks_" + std::to_string(size_));
    }

    if (serial) {
        delete serial;
        serial = NULL;
    }
    if (serial_rosen) {
        delete serial_rosen;
        serial_rosen = NULL;
    }
}

template <typename T>
void good_mpi_ga_exps(const ExperimentConfigs config) {
    int MAX_ITERS, minSize, maxSize, mult;
    int rank_, size_;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

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
                            auto results = mpi_ga_rosen(problem_size, seed, size_, num_iters_method, population_size, 10);

                            if (!rank_)
                                handler.get_result_row(get_result_row<float>(results, size_, -1, -1, seed, problem, problem_size, num_iters_method, results.result_filename, "GenAlg", population_size));
                        } else {
                            auto results = mpi_ga_tsp(problem_size, seed, size_, num_iters_method, population_size, 10);

                            if (!rank_)
                                handler.get_result_row(get_result_row<int>(results, size_, -1, -1, seed, problem, problem_size, num_iters_method, results.result_filename, "GenAlg", population_size));
                        }
                    }
                }
            }
        }
    }
    if (config.which_exp == 2) {
        printf("GenAlg, exp %d, problem = %s\n", config.which_exp, problem.c_str());
        directory = "ga_exp2_weak";
        for (int num_iters_method = 100; num_iters_method <= 1000; num_iters_method *= 10) {
            for (int problem_size = minSize; problem_size <= maxSize; problem_size *= mult) {
                for (int population_size = 28; population_size <= 28; population_size *= 2) {
                    for (int seed = 42; seed < 42 + 3; ++seed) {
                        printf("\rIters %d Prob %d Seed %d Pop Size %d                          ",
                               num_iters_method, problem_size, seed, population_size);
                        auto pop_size_to_do = population_size * size_;
                        printf("Pop size = %d\n", pop_size_to_do);
                        std::string row = "";
                        if (config.is_rosen) {
                            auto results = mpi_ga_rosen(problem_size, seed, size_, num_iters_method, pop_size_to_do, 10);
                            if (!rank_)
                                row = get_result_row(results, size_, -1, -1, seed, "TSP", problem_size, num_iters_method, results.result_filename, "GenAlg", pop_size_to_do);
                        } else {
                            auto results = mpi_ga_tsp(problem_size, seed, size_, num_iters_method, pop_size_to_do, 10);
                            if (!rank_)
                                row = get_result_row(results, size_, -1, -1, seed, "TSP", problem_size, num_iters_method, results.result_filename, "GenAlg", pop_size_to_do);
                        }
                        if (!rank_)
                            handler.get_result_row(row);
                    }
                }
            }
        }
    }
    printf("\n");
    if (should_write && rank_ == 0) {
        handler.to_file("proper_exps/" + VERSION + "/mpi/ga/" + directory + "_" + problem, "ranks_" + std::to_string(size_));
    }

    if (serial_salesman_pop) {
        delete serial_salesman_pop;
        serial_salesman_pop = NULL;
    }
}

void mpi_run_all_experiments_according_to_args(int argc, char **argv) {
    const ExperimentConfigs config = parse_all_experiment_args(argc, argv);
    if (config.is_sa) {
        if (config.is_rosen)
            good_mpi_sa_exps<float>(config);
        else
            good_mpi_sa_exps<int>(config);
    } else {
        if (config.is_rosen)
            good_mpi_ga_exps<float>(config);
        else
            good_mpi_ga_exps<int>(config);
    }
}

void master_slave_ga() {
    Random::seed(42);
    int C = 1000;
    auto prob = createProblem(C, -10, 10);
    int pop_size = 256;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto solver = MPIMasterSlavePop(pop_size, prob);
    auto begin = std::chrono::high_resolution_clock::now();
    solver.init();
    Random::seed(rank * 500);
    solver.solveProblem(10000);
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - begin).count();
    MPI_Barrier(MPI_COMM_WORLD);

    if (!rank)
        printf("Did in %fms\n", time);
}

Results<int> mpi_ga_tsp(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Random::seed(seed);
    const TSP prob = createProblem(problem_size, -10, 10);
    auto results = mpi_do_all_gen_alg<int, TSP>(
        prob, [](int problem_size, const TSP &prob, int numranks, int population_size) {
            if (serial_salesman_pop) {
                delete serial_salesman_pop;
                serial_salesman_pop = NULL;
            }
            serial_salesman_pop = new SalesmanPopulation(population_size, prob.C, prob);
            serial_salesman_pop->score_push_back = false;
            return new MPIPopulation<int>(serial_salesman_pop);
        },
        problem_size, seed, ranks, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, prob.description());
    return results;
}

Results<int> mpi_sa_tsp(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Results<int> res;
    Random::seed(seed);
    const TSP prob = createProblem(problem_size, -10, 10);
    auto results = mpi_do_all_sa<int, TSP>(
        prob, [](int problem_size, const TSP &prob, int numranks) {
            if (serial) {
                delete serial;
                serial = NULL;
            }
            serial = new SimulatedAnnealingSalesman(prob.C, prob);
            serial->score_push_back = false;
            return new MPIAnnealing<int>(serial);
        },
        problem_size, seed, ranks, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, prob.description());
    return results;
}

Results<float> mpi_sa_rosen(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Results<float> res;
    Random::seed(seed);
    const Rosenbrock prob = Rosenbrock{problem_size};
    auto results = mpi_do_all_sa<float, Rosenbrock>(
        prob, [](int problem_size, const Rosenbrock &prob, int numranks) {
            if (serial_rosen) {
                delete serial_rosen;
                serial_rosen = NULL;
            }
            serial_rosen = new SimulatedAnnealingRosenbrock(problem_size, prob);
            serial_rosen->score_push_back = false;
            return new MPIAnnealing<float>(serial_rosen);
        },
        problem_size, seed, ranks, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, "Rosen" + std::to_string(problem_size));
    return results;
}

Results<float> mpi_ga_rosen(int problem_size, int seed, int ranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Random::seed(seed);
    const Rosenbrock prob = {problem_size};
    auto results = mpi_do_all_gen_alg<float, Rosenbrock>(
        prob, [](int problem_size, const Rosenbrock &prob, int numranks, int population_size) {
            if (serial_salesman_pop_rosen) {
                delete serial_salesman_pop_rosen;
                serial_salesman_pop_rosen = NULL;
            }
            serial_salesman_pop_rosen = new RosenbrockPopulation(population_size, problem_size, prob);
            serial_salesman_pop_rosen->score_push_back = false;
            return new MPIPopulation<float>(serial_salesman_pop_rosen);
        },
        problem_size, seed, ranks, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, "Rosen_" + std::to_string(problem_size));
    return results;
}

template <typename T, typename Problem>
Results<T> mpi_do_all_sa(const Problem &problem, mpi_get_annealing_pointer<T, Problem> get_annealing_ptr,
                         int problem_size, int seed, int numranks, int num_iterations_for_method, int pop_size,
                         int num_iters_for_single_run_experiment, const std::string &prob_desc) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto func = [&](int iter_number) {
        Results<T> res;
        Random::seed(seed * 100 + rank + iter_number * 5000);

        const Problem &prob = problem;

        auto begin = std::chrono::high_resolution_clock::now();
        MPIAnnealing<T> *pop_ = get_annealing_ptr(problem_size, prob, numranks);
#if DO_SERIAL
        // if serial, take the serial class, to have a comparison with the serial implementation.
        auto pop = pop_->serial_class;
        pop->score_push_back = true;
#else
        auto pop = pop_;
#endif
        pop->init();
        auto end = std::chrono::high_resolution_clock::now();

        res.time_ms_to_initialise = std::chrono::duration<double, std::milli>(end - begin).count();
        begin = std::chrono::high_resolution_clock::now();
        pop->solveProblem(num_iterations_for_method);

        MPI_Barrier(MPI_COMM_WORLD);
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
#ifdef DEMO
        if (rank == 0) {
            printf("\tTime taken = %lfs. Score at end = %f. Result is %s\n", d / 1e6, res.scores.back(), is_valid ? "Valid" : "Invalid");
        }
#endif

#if DO_SERIAL
        delete pop_;
#else
        delete pop;
#endif
        return res;
    };
    Experiment<T> e("MPI, SA.\n" + prob_desc, "seeds_" + std::to_string(seed), func);
    auto pos = "proper_exps/" + VERSION + "/mpi/" + directory + "/" + date + "_" + std::to_string(numranks) + "/";
    if (write_to_text_file && rank == 0){
        int _ = system(("mkdir -p " + pos).c_str());
    }
    auto results = e.run(pos, num_iters_for_single_run_experiment, write_to_text_file && rank == 0);
    return results;
}

template <typename T, typename Problem>
Results<T> mpi_do_all_gen_alg(const Problem &problem, mpi_get_population_pointer<T, Problem> get_pop_pointer, int problem_size, int seed, int numranks, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment, const std::string &prob_desc) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto func = [&](int iter_number) {
        Results<T> res;
        Random::seed(seed * 100 + rank + iter_number * 5000);

        const Problem &prob = problem;
        auto begin = std::chrono::high_resolution_clock::now();
        MPIPopulation<T> *pop = get_pop_pointer(problem_size, prob, numranks, pop_size);

        pop->init();

        auto end = std::chrono::high_resolution_clock::now();
        res.time_ms_to_initialise = std::chrono::duration<double, std::milli>(end - begin).count();

        begin = std::chrono::high_resolution_clock::now();
        pop->solveProblem(num_iterations_for_method);
        MPI_Barrier(MPI_COMM_WORLD);
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
    Experiment<T> e("MPI, GA.\n" + prob_desc, "seeds_" + std::to_string(seed), func);
    auto pos = "proper_exps/" + VERSION + "/mpi/ga/" + directory + "/" + date + "_" + std::to_string(numranks) + "/";
    if (write_to_text_file && rank == 0)
        int _ = system(("mkdir -p " + pos).c_str());
    auto results = e.run(pos, num_iters_for_single_run_experiment, write_to_text_file && rank == 0);
    return results;
}
