#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__
#include <string>
#include <algorithm>
#include <vector>
#include <functional>

/**
 * @brief Struct representing results for the execution of experiments.
 * 
 * @tparam T 
 */
template <typename T>
struct Results {
    double time_ms_to_initialise = 0;
    double time_ms_to_perform_operations = 0;
    std::vector<float> scores;
    // The final individual.
    std::vector<T> final_result;

    int num_iterations = 0;
    
    // equal to 1 for serial, nranks for MPI and blockDim.x * gridDim.x for cuda
    int num_procs = 0;

    long long total_num_operations = 0;
    
    std::string result_filename = "";

    void operator+=(const Results<T>& other);
    
    void operator/=(const double& other);

    std::string to_string() const;
    ~Results(){
    }
};


// Function that returns a Results<T> and takes in an int.
template <typename T>
using run_single_iteration = std::function<Results<T>(int)>;

/**
 * @brief Represents a single experiment, consisting of a function to run, for how many iterations, where results should be stored, etc.
 * 
 * @tparam T 
 */
template <typename T>
struct Experiment{
   std::string name_description, file_suffix;
   run_single_iteration<T> function_to_run;
   std::string saved_filename = "";
   Experiment(std::string desc, std::string _file_suffix, run_single_iteration<T> _function_to_run): name_description(desc), file_suffix(_file_suffix), function_to_run(_function_to_run){}

   Results<T> run(std::string directory_name, int number_of_iterations = 10, bool should_write=true);

   static std::string results_to_file(const Results<T> &result, std::string directory_name, std::string description, std::string file_suffix, int number_of_iterations, std::string &result_filename);
};

/**
 * @brief Returns a string that represents a CSV row, with all the pertinent information.
 * If something isn't valid (like pop_size in SA), simply give -1.
 * 
 * @tparam T 
 * @param results 
 * @param mpi_nodes 
 * @param cuda_block_size 
 * @param cuda_grid_size 
 * @param seed 
 * @param problem_name 
 * @param problem_size 
 * @param num_iterations 
 * @param result_filename 
 * @param method 
 * @param pop_size 
 * @return std::string 
 */
template <typename T>
std::string get_result_row(const Results<T> &results, int mpi_nodes, int cuda_block_size, int cuda_grid_size, int seed, std::string problem_name, int problem_size, int num_iterations, const std::string &result_filename, std::string method="SA", int pop_size=-1);

std::string combine(std::vector<std::string> strings);

/**
 * @brief Writes the final results to a .csv file for later analysis.
 * 
 * @tparam T 
 */
template <typename T>
struct CSVHandler{
    std::string current_csv_string = "";
    std::string header_string = "";
    CSVHandler(){
        header_string = get_header_string();
        current_csv_string = header_string + "\n";
    }


    std::string get_header_string();

    std::string get_result_row(const Results<T> &results, int mpi_nodes, int cuda_block_size, int cuda_grid_size, int seed, std::string problem_name, int problem_size, int num_iterations, const std::string &result_filename, std::string method="SA", int pop_size=-1);
    std::string get_result_row(const std::string &row);

    void to_file(std::string directory, std::string suffix);
};

#endif // __EXPERIMENT_H__