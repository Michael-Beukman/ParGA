#include "common/experiments/Experiment.h"

#include <assert.h>

#include <sstream>

#include "common/metrics/metrics.h"

template <typename T>
Results<T> Experiment<T>::run(std::string directory_name, int number_of_iterations, bool should_write) {
    assert(number_of_iterations > 1);
    Results<T> cumulative_results;
    for (int i = 0; i < number_of_iterations; ++i) {
        auto current_results = function_to_run(i);
        // warm up and ignore the first run.
        if (i == 0) continue;
        cumulative_results += current_results;
    }
    cumulative_results /= (double)(number_of_iterations - 1);
    if (should_write)
        cumulative_results.result_filename = results_to_file(cumulative_results, directory_name, name_description, file_suffix, number_of_iterations, saved_filename);
    return cumulative_results;
}
template <typename T>

std::string Results<T>::to_string() const {
    std::stringstream ss;
    ss << "Version: 1\n";
    ss << "time_init_ms: " << time_ms_to_initialise << "\n";
    ss << "time_ops_ms: " << time_ms_to_perform_operations << "\n";
    ss << "num_iterations: " << num_iterations << "\n";
    ss << "num_procs: " << num_procs << "\n";
    ss << "num_ops: " << total_num_operations << "\n";
    ss << "scores:\n";
    for (auto score : scores)
        ss << score << " ";
    ss << "\n";
    ss << "Best Result:\n[ ";
    for (auto num : final_result)
        ss << num << " ";
    ss << "]\n";
    return ss.str();
}

template <typename T>
std::string Experiment<T>::results_to_file(const Results<T>& result, std::string directory_name, std::string description, std::string file_suffix, int number_of_iterations, std::string& result_filename) {
    auto result_string = result.to_string();
    std::stringstream ss;
    auto date = get_date();

    ss << "Date: " << date << "\n";
    ss << "Description: " << description << "\n";
    ss << "Num Runs: " << number_of_iterations << "\n";
    ss << result_string;
    result_filename = saveStringToFile(ss.str(), directory_name, file_suffix);
    return result_filename;
}

/**
 * @brief Adds two results.
 * 
 * @tparam T 
 * @param other 
 */
template <typename T>
void Results<T>::operator+=(const Results<T>& other) {
    time_ms_to_initialise += other.time_ms_to_initialise;
    time_ms_to_perform_operations += other.time_ms_to_perform_operations;
    num_iterations += other.num_iterations;
    num_procs += other.num_procs;
    total_num_operations += other.total_num_operations;

    // now scores
    if (scores.size() == 0) {
        scores = other.scores;
    } else {
        std::transform(scores.begin(), scores.end(), other.scores.begin(), scores.begin(), std::plus<float>());
    }
    final_result = other.final_result;
}

/**
 * @brief Divides by a double, usually the number of (runs - 1) to average accross.
 * 
 * @tparam T 
 * @param other 
 */
template <typename T>
void Results<T>::operator/=(const double& other) {
    time_ms_to_initialise /= other;
    time_ms_to_perform_operations /= other;
    num_iterations /= other;
    num_procs /= other;
    total_num_operations /= other;

    std::transform(scores.begin(), scores.end(), scores.begin(), [&](float val) {
        return val / (other);
    });
}

std::string combine(std::vector<std::string> strings) {
    std::stringstream ss;
    char buffer[1000];
    for (int i = 0; i < strings.size(); ++i) {
        if (i == strings.size() - 1)
            sprintf(buffer, "%-50s", strings[i].c_str());
        else
            sprintf(buffer, "%-50s,", strings[i].c_str());
        ss << buffer;
    }
    return ss.str();
}
/**
 * @brief The header for the csv file.
 * 
 * @tparam T 
 * @return std::string 
 */
template <typename T>
std::string CSVHandler<T>::get_header_string() {
    return combine(
        {"Method", "Problem", "ProbSize", "Seed", "Time Init", "Time Ops", "Num Ops", "Init Score", "Final Score", "Procs", "MPI_Nodes", "CUDA_BlockSize", "CUDA_GridSize", "Num Iterations", "PopSize", "ResultsFilename"});
}

template <typename T>
std::string CSVHandler<T>::get_result_row(const Results<T>& results, int mpi_nodes, int cuda_block_size, int cuda_grid_size, int seed, std::string problem_name, int problem_size, int num_iterations,
                                          const std::string& result_filename, std::string method, int pop_size) {
    auto t = [](int i) { return std::to_string(i); };
    auto td = [](double i) { return std::to_string(i); };
    auto row = combine(
        {method,
         problem_name,
         t(problem_size),
         t(seed),
         td(results.time_ms_to_initialise),
         td(results.time_ms_to_perform_operations),
         t(results.total_num_operations),
         td(results.scores.front()),
         td(results.scores.back()),
         t(results.num_procs),
         t(mpi_nodes),
         t(cuda_block_size),
         t(cuda_grid_size),
         t(num_iterations),
         t(pop_size),
         result_filename});
    current_csv_string += row + "\n";
    return row;
}

template <typename T>
std::string CSVHandler<T>::get_result_row(const std::string &row){
    current_csv_string += row + "\n";
    return row;
}
template <typename T>
void CSVHandler<T>::to_file(std::string directory, std::string suffix) {
    // saveStringToFile("", directory, suffix, ".csv");
    saveStringToFile(current_csv_string, directory, suffix, ".csv");
}

template <typename T>
std::string get_result_row(const Results<T>& results, int mpi_nodes, int cuda_block_size, int cuda_grid_size, int seed, std::string problem_name, int problem_size, int num_iterations, const std::string& result_filename, std::string method, int pop_size) {
    auto t = [](int i) { return std::to_string(i); };
    auto td = [](double i) { return std::to_string(i); };
    auto row = combine(
        {
        method,
         problem_name,
         t(problem_size),
         t(seed),
         td(results.time_ms_to_initialise),
         td(results.time_ms_to_perform_operations),
         t(results.total_num_operations),
         td(results.scores.front()),
         td(results.scores.back()),
         t(results.num_procs),
         t(mpi_nodes),
         t(cuda_block_size),
         t(cuda_grid_size),
         t(num_iterations),
         t(pop_size),
         result_filename});
    return row;
}

template std::string get_result_row<int>(const Results<int>& results, int mpi_nodes, int cuda_block_size, int cuda_grid_size, int seed, std::string problem_name, int problem_size, int num_iterations, const std::string& result_filename, std::string method, int pop_size);
template std::string get_result_row<float>(const Results<float>& results, int mpi_nodes, int cuda_block_size, int cuda_grid_size, int seed, std::string problem_name, int problem_size, int num_iterations, const std::string& result_filename, std::string method, int pop_size);


template class Experiment<int>;
template class Results<int>;
template class Results<float>;
template class Experiment<float>;
template class CSVHandler<float>;
template class CSVHandler<int>;