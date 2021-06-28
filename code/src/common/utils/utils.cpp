#include "common/utils/utils.h"
/**
 * @brief Simple function to parse the demo arguments with regards to how many iterations and how the scaling should work.
 * 
 * @param argc 
 * @param argv 
 * @param ranks 
 * @return std::tuple<int, int, std::string> Total Num Iterations (1000000), iterations per rank, scaling string to output
 */
std::tuple<int, int, std::string> parse_demo_args(int argc, char** argv, int ranks){
    const int og = 1000000;
    int iterations = og;
    int scaling = 0;
    std::string scaling_string = "None";
    if (argc == 1){
        // just program
    }else if (argc == 2){
        // program and num iterations
        iterations = atoi(argv[1]);
    }else {
        // program, iterations & scaling method
        iterations = atoi(argv[1]);
        if (std::string(argv[2]) == "NONE"){
            scaling = 0;
            // no change to iterations
        }else if (std::string(argv[2]) == "FULL"){
            scaling = 1;
            scaling_string = "Full";
            iterations /= ranks;
            // 
        }else{
            scaling = atoi(argv[2]);
            scaling_string = "Custom (" + std::string(argv[2]) +")";
            iterations = scaling * iterations / ranks;
        }
    }

    return {og, iterations, scaling_string};

}

ExperimentConfigs parse_all_experiment_args(int argc, char** argv){
    ExperimentConfigs config {true, 1, false};
    if (argc != 4){
        printf("Invalid number of arguments. Need 4, received %d\n", argc);
        exit(1);
    }
    char* sa = argv[1];
    char* which_exp = argv[2];
    char* rosen = argv[3];
    config.is_sa = atoi(sa);
    config.which_exp = atoi(which_exp);
    config.is_rosen = atoi(rosen);
    printf("Running experiment %s + exp %d + %s\n", config.is_sa ? "SA" : "GA", config.which_exp, config.is_rosen ? "Rosen" : "TSP");
    return config;
}