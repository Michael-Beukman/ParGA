#include <assert.h>

#include <chrono>
#include <functional>
#include <type_traits>

#include "CUDA/problems/rosenbrock/ga/CudaRosenbrockPopulation.h"
#include "CUDA/problems/rosenbrock/sa/CudaAnnealingRosenbrock.h"
#include "CUDA/problems/salesman/ga/CudaSalesmanPop.h"
#include "CUDA/problems/salesman/annealing/CudaAnnealingSalesman.h"
#include "common/experiments/Experiment.h"
#include "common/metrics/metrics.h"
#include "common/problems/rosenbrock/rosenbrock.h"
#include "common/problems/salesman/SalesmanUtils.h"
#include "common/utils/utils.h"
#include "common/utils/random/Random.h"
#ifdef DEMO
const bool should_write = false;
#define EXP_VERBOSE 0
#else 
const bool should_write = true;
#define EXP_VERBOSE 0
#endif

#ifdef LOG_ALL
const bool write_to_text_file = should_write;
#else
const bool write_to_text_file = false;
#endif

std::string directory = "test_v1";
const std::string VERSION = "v50_cuda";
/**
 * @brief Runs experiments for the cuda code.
 *          It runs GA & SA on both TSP and Rosenbrock.
 *          Also performs a grid search type thing over parameters (e.g. problem size, blockDim, gridDim, num_iterations, etc)
 */
void run_all_experiments();
std::string date = get_date();
template <typename T>
using run_single_experiment = void (*)(int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);

template <typename T, typename Problem>
using get_annealing_pointer = std::function<CudaSimulatedAnnealing<T> *(int problem_size, const Problem &prob, int block_size, int grid_size)>;
template <typename T, typename Problem>
using get_ga_pointer = std::function<CudaPopulation<T> *(int problem_size, const Problem &prob, int population_size, int block_size, int grid_size)>;

Results<int> cuda_ga_tsp(int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);
Results<int> cuda_sa_tsp(int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);
Results<float> cuda_ga_rosen(int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);
Results<float> cuda_sa_rosen(int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment);

template <typename T, typename Problem>
Results<T> cuda_do_all_sa(const Problem &problem, get_annealing_pointer<T, Problem> get_annealing_ptr, int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment, const std::string &prob_desc);

template <typename T, typename Problem>
Results<T> cuda_do_all_genetic_alg(const Problem &problem, get_ga_pointer<T, Problem> get_population, int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment, const std::string &prob_desc);

void cuda_run_all_experiments_according_to_args(int argc, char **argv);
template <typename T>
void good_cuda_ga_exps(ExperimentConfigs config);
template <typename T>
void good_cuda_sa_exps(ExperimentConfigs config);

void demo(int argc, char **argv) {
    int C = 1000;
    auto iters_string = parse_demo_args(argc, argv, 1);

    int num_iters_total = std::get<0>(iters_string);
    int num_iters = std::get<1>(iters_string);
    auto str = std::get<2>(iters_string);
    printf("Running TSP with total number of iterations = %d, scaling = %s. (CUDA)\n", num_iters, str.c_str());

    auto results = cuda_sa_tsp(C, 0, 4, 256, num_iters, -1, 2);
}

void simple_tests() {
    int C = 1000;
    auto results = cuda_sa_tsp(C, 0, 4, 128, 1000, -1, 2);
}

int main(int argc, char **argv) {
#ifdef DEMO
    demo(argc, argv);
#else
    cuda_run_all_experiments_according_to_args(argc, argv);
#endif
}

template <typename T>
void good_cuda_ga_exps(ExperimentConfigs config) {
    int minSize, maxSize, mult;
    CSVHandler<T> handler;
    std::string problem;
    if (config.is_rosen) {
        problem = "Rosen";
        minSize = 2;
        maxSize = 64;
        mult = 2;
    } else {
        problem = "TSP";
        minSize = 10;
        maxSize = 1000;
        mult = 10;
    }

    int which_experiment = 1;
    if (which_experiment == 1) {
        directory = "cuda_ga_experiment_1_standard_" + problem;
        printf("Doing experiment 1 for CUDA Genetic Alg with prob = %s\n", problem.c_str());
        for (int num_iters_method = 10; num_iters_method <= 5120; num_iters_method *= 2) {
            for (int population_size = 64; population_size <= 1024; population_size *= 2) {
                // for cuda GA, population size determines block and gridSize.
                int blockSize = -1;
                int gridSize = -1;
                for (int problem_size = minSize; problem_size <= maxSize; problem_size *= mult) {
                    for (int seed = 42; seed < 42 + 3; ++seed) {
                        printf("Iters %d ProbSize %d PopulationSize %d Seed %d\n",
                               num_iters_method, problem_size, population_size, seed);
                        fflush(stdout);
                        std::string row = "";
                        if (config.is_rosen) {
                            auto results = cuda_ga_rosen(problem_size, seed, blockSize, gridSize, num_iters_method, population_size, 10);
                            row = get_result_row(results, -1, blockSize, gridSize, seed, "Rosen", problem_size, results.num_iterations, results.result_filename, "CUDA_GA", population_size);
                        } else {
                            auto results = cuda_ga_tsp(problem_size, seed, blockSize, gridSize, num_iters_method, population_size, 10);
                            row = get_result_row(results, -1, blockSize, gridSize, seed, "TSP", problem_size, results.num_iterations, results.result_filename, "CUDA_GA", population_size);
                        }
                        handler.get_result_row(row);
                    }
                }
            }
        }
    }
    printf("\n");
    handler.to_file("proper_exps/" + VERSION + "/cuda/ga/" + directory, "");
}

template <typename T>
void good_cuda_sa_exps(ExperimentConfigs config) {
    int minSize, maxSize, mult;
    CSVHandler<T> handler;
    std::string problem;
    if (config.is_rosen) {
        problem = "Rosen";
        minSize = 2;
        maxSize = 64;
        mult = 2;
    } else {
        problem = "TSP";
        minSize = 10;
        maxSize = 1000;
        mult = 10;
    }

    if (config.which_exp == 1) {
        printf("Doing experiment 1 for CUDA SA with prob = %s.\n", problem.c_str()); fflush(stdout);
        directory = "cuda_sa_experiment4_all" + problem;
        std::vector<int> iterations = {10, 100, 500, 1000, 1500, 2000, 2500, 3000, 5000, 7500};
        for (int num_iters_method : iterations) {
            for (int blockSize = 4; blockSize <= 128; blockSize *= 2) {
                for (int gridSize = 16; gridSize <= 1024; gridSize *= 2) {
                    if (blockSize >= 128 && gridSize >= 512) continue; // too large.
                    for (int problem_size = minSize; problem_size <= maxSize; problem_size *= mult) {
                        for (int seed = 42; seed < 42 + 3; ++seed) {
                            printf("Iters %d BlockSize %d GridSize %d ProbSize %d Seed %d\n",
                                   num_iters_method, blockSize, gridSize, problem_size, seed);
                            fflush(stdout);

                            std::string row = "";

                            if (config.is_rosen) {
                                auto results = cuda_sa_rosen(problem_size, seed, blockSize, gridSize, num_iters_method, -1, 10);
                                row = get_result_row(results, -1, blockSize, gridSize, seed, problem, problem_size, num_iters_method, results.result_filename, "CUDA_SA");
                            } else {
                                auto results = cuda_sa_tsp(problem_size, seed, blockSize, gridSize, num_iters_method, -1, 10);
                                row = get_result_row(results, -1, blockSize, gridSize, seed, problem, problem_size, num_iters_method, results.result_filename, "CUDA_SA");
                            }
                            handler.get_result_row(row);
                        }
                    }
                }
            }
        }
    }

    printf("\n");
    handler.to_file("proper_exps/" + VERSION + "/cuda/sa/" + directory, "");
}

void cuda_run_all_experiments_according_to_args(int argc, char **argv) {
    const ExperimentConfigs config = parse_all_experiment_args(argc, argv);
    if (config.is_sa) {
        if (config.is_rosen)
            good_cuda_sa_exps<float>(config);
        else
            good_cuda_sa_exps<int>(config);
    } else {
        if (config.is_rosen)
            good_cuda_ga_exps<float>(config);
        else
            good_cuda_ga_exps<int>(config);
    }
}

Results<float> cuda_sa_rosen(int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Results<int> res;
    Random::seed(seed);

    Rosenbrock prob = {problem_size};
    auto results = cuda_do_all_sa<float, Rosenbrock>(
        prob, [](int problem_size, const Rosenbrock &prob, int block_size, int grid_size) {
            return new CudaAnnealingRosenbrock(problem_size, prob, block_size, grid_size);
        },
        problem_size, seed, blockDim, gridDim, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, "Rosenbrock Size " + std::to_string(prob.N));
    return results;
}

Results<int> cuda_sa_tsp(int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Results<int> res;
    Random::seed(seed);
    const TSP prob = createProblem(problem_size, -10, 10);
    auto results = cuda_do_all_sa<int, TSP>(
        prob, [](int problem_size, const TSP &prob, int block_size, int grid_size) {
            return new CudaAnnealingSalesman(problem_size, prob, block_size, grid_size);
        },
        problem_size, seed, blockDim, gridDim, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, prob.description());
    return results;
}

Results<float> cuda_ga_rosen(int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Random::seed(seed);
    const Rosenbrock prob = {problem_size};
    auto results = cuda_do_all_genetic_alg<float, Rosenbrock>(
        prob, [](int problem_size, const Rosenbrock &prob, int population_size, int block_size, int grid_size) {
            return new CudaRosenbrockPop(population_size, problem_size, prob);
        },
        problem_size, seed, blockDim, gridDim, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, "RosenBrock N = " + std::to_string(problem_size));
    return results;
}

Results<int> cuda_ga_tsp(int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment) {
    Random::seed(seed);
    const TSP prob = createProblem(problem_size, -10, 10);
    auto results = cuda_do_all_genetic_alg<int, TSP>(
        prob, [](int problem_size, const TSP &prob, int population_size, int block_size, int grid_size) {
            return new CudaSalesmanPop(population_size, problem_size, prob);
        },
        problem_size, seed, blockDim, gridDim, num_iterations_for_method, pop_size, num_iters_for_single_run_experiment, prob.description());
    return results;
}

template <typename T, typename Problem>
Results<T> cuda_do_all_sa(const Problem &problem, get_annealing_pointer<T, Problem> get_annealing_ptr, int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int pop_size, int num_iters_for_single_run_experiment,
                          const std::string &prob_desc) {
    auto func = [&](int iter_number) {
        Results<T> res;
        Random::seed(seed);

        const Problem &prob = problem;
        auto begin = std::chrono::high_resolution_clock::now();
        CudaSimulatedAnnealing<T> *pop = get_annealing_ptr(problem_size, prob, blockDim, gridDim);

        pop->init();
        auto end = std::chrono::high_resolution_clock::now();
        res.time_ms_to_initialise = std::chrono::duration<double, std::milli>(end - begin).count();

        begin = std::chrono::high_resolution_clock::now();
        pop->solveProblem(num_iterations_for_method);
        end = std::chrono::high_resolution_clock::now();
        res.time_ms_to_perform_operations = std::chrono::duration<double, std::milli>(end - begin).count();

        double time_in_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        double d = time_in_ms * 1000.0;
        long long num_ops = get_number_of_operations(prob, num_iterations_for_method, NUM_ITERS_IN_BLOCK * pop->gridDim.x * pop->blockDim.x);
        res.scores = pop->get_all_measured_scores();
        res.num_iterations = num_iterations_for_method * NUM_ITERS_IN_BLOCK;
        res.num_procs = pop->blockDim.x * pop->gridDim.x;
        res.total_num_operations = num_ops;

        res.final_result = pop->get_final_best_solution();
        bool is_valid = is_individual_valid(res.final_result.data(), prob);
        assert(1 == is_valid);
#if EXP_VERBOSE
        printf("Time taken = %lfs for %lld ops, and %'dM gen_pop per second. Score at end = %f. Is valid = %i\n", d / 1e6, num_ops, (int)((num_ops) / d), res.scores.back(), is_valid);
#endif
#ifdef DEMO
        printf("\tTime taken = %lfs. Score at end = %f. Result is %s\n", d / 1e6, res.scores.back(), is_valid ? "Valid" : "Invalid");
#endif
        delete pop;
        return res;
    };
    Experiment<T> e("CUDA, SA.\n" + prob_desc, "seeds_" + std::to_string(seed), func);
    std::string pos = "/tmp";
    if (write_to_text_file && should_write) {
        pos = "proper_exps/" + VERSION + "/cuda/sa/" + directory + "/" + date + "/";
        system(("mkdir -p " + pos).c_str());
    }
    auto results = e.run(pos, num_iters_for_single_run_experiment, write_to_text_file && should_write);
    return results;
}

template <typename T, typename Problem>
Results<T> cuda_do_all_genetic_alg(const Problem &problem, get_ga_pointer<T, Problem> get_population, int problem_size, int seed, int blockDim, int gridDim, int num_iterations_for_method, int population_size, int num_iters_for_single_run_experiment, const std::string &prob_desc) {
    auto func = [&](int iter_number) {
        Results<T> res;
        Random::seed(seed);

        auto begin = std::chrono::high_resolution_clock::now();
        CudaPopulation<T> *pop = get_population(problem_size, problem, population_size, blockDim, gridDim);

        // init stuff
        pop->init();
        auto end = std::chrono::high_resolution_clock::now();
        res.time_ms_to_initialise = std::chrono::duration<double, std::milli>(end - begin).count();

        begin = std::chrono::high_resolution_clock::now();
        pop->solveProblem(num_iterations_for_method);
        end = std::chrono::high_resolution_clock::now();
        res.time_ms_to_perform_operations = std::chrono::duration<double, std::milli>(end - begin).count();

        double time_in_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        double d = time_in_ms * 1000.0;
        long long num_ops = get_number_of_operations(problem, num_iterations_for_method, population_size);
        res.scores = pop->get_all_measured_scores();
        res.num_iterations = num_iterations_for_method;
        res.num_procs = pop->dimBlock.x * pop->dimGrid.x;
        res.total_num_operations = num_ops;
        res.final_result = pop->get_final_best_solution();
        bool is_valid = is_individual_valid(res.final_result.data(), problem);
        assert(1 == is_valid);
#if EXP_VERBOSE
        printf("Time taken = %lfms, and %lld M gen_pop per second. At end = %f\n", time_in_ms, (long long)((num_ops) / d), res.scores.back());
#endif
        delete pop;
        return res;
    };

    Experiment<T> e("CUDA, GA.\n" + prob_desc, "seeds_" + std::to_string(seed), func);
    std::string pos = "/tmp";
    if (write_to_text_file && should_write) {
        pos = "proper_exps/" + VERSION + "/cuda/ga/" + directory + "/" + date + "/";
        system(("mkdir -p " + pos).c_str());
    }
    auto results = e.run(pos, num_iters_for_single_run_experiment, write_to_text_file && should_write);

    return results;
}
