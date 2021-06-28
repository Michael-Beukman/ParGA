#ifndef __UTILS_H__
#define __UTILS_H__
#include  <string>
#include  <tuple>

/**
 * @brief Returns a string representing a single individual
 * 
 * @tparam T 
 * @param C 
 * @param individual 
 * @return std::string 
 */
template <typename T>
std::string printSingle(int C, T* individual) {
    std::string ans = "";
    char buffer[1024];
    sprintf(buffer, "[");
    ans += buffer;
    for (int i = 0; i < C; ++i) {
        sprintf(buffer, "%d", individual[i]);
        ans += buffer;
        if (i != C - 1) {
            sprintf(buffer, ", ");
            ans += buffer;
        }
    }
    sprintf(buffer, "]");
    ans += buffer;

    return ans;
}

template <typename T>
void _printPop(T* pop, int N, int C) {
    for (int i = 0; i < N; ++i) {
        printf("\t%s\n", printSingle(C, pop + C * i).c_str());
    }
}

// number of iterations
// Scaling can be
//      0: No scaling, same number of iterations
//      1: Full scaling, iters / ranks;
//      Any number greater than this iters * num / ranks;
std::tuple<int, int, std::string> parse_demo_args(int argc, char** argv, int ranks);

/**
 * @brief A useful data class that contains some info about which experiment to run.
 * 
 */
struct ExperimentConfigs{
    bool is_sa;
    int which_exp;
    bool is_rosen;
};
ExperimentConfigs parse_all_experiment_args(int argc, char** argv);
#endif // __UTILS_H__