#include "common/metrics/metrics.h"
long long get_number_of_operations(const TSP& problem, int number_of_steps, int num_evals_per_step) {
    return problem.C * (long long)number_of_steps * (long long)num_evals_per_step;
}
long long get_number_of_operations(const Rosenbrock& problem, int number_of_steps, int num_evals_per_step){
    // Very rough number of operations.
    // we have a few main things.
    // First of all, evaluations. Each evaluation takes up N * (5 mults, 3 adds) => 8 FLOP
    return 8 * problem.N * (long long)number_of_steps * (long long)num_evals_per_step;
}

std::string get_date() {
    // returns a pretty formatted date.
    std::time_t rawtime;
    std::tm* timeinfo;
    char buffer[80];

    std::time(&rawtime);
    timeinfo = std::localtime(&rawtime);
    std::strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", timeinfo);
    return std::string(buffer);
}

std::string saveStringToFile(std::string s, std::string subfolder, std::string file_suffix, std::string file_extension) {
    auto name = get_date() + "__" + file_suffix;
    auto folder = "results/" + subfolder + "/";
    int ans = system(("mkdir -p " + folder).c_str());
    if (ans == -1) {
        printf("Error occurred in create dirs\n");
    }
    std::ofstream file;
    std::string all_name = folder + name + file_extension;
    
    file.open(all_name);
    file << s;
    file.close();
    return all_name;
}
std::string getResultsString(const std::vector<float> scores, double time_in_milliseconds, double num_ops, std::string description) {
    auto s = "Description: " + description + "\nNum_ops: " + std::to_string(num_ops) + "\nTime_ms: " + std::to_string(time_in_milliseconds) + "\n";
     for (auto score : scores) {
        s += std::to_string(score) + "\n";
    }
    return s;
}