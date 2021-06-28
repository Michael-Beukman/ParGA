#ifndef __METRICS_H__
#define __METRICS_H__

#include <vector>
#include <utility>
#include <string>
#include <chrono>
#include <fstream>
#include "common/problems/salesman/TSP.h"
#include "common/problems/rosenbrock/rosenbrock.h"

// Some useful metrics and utility functions.

// These are actually not really used.
long long get_number_of_operations(const TSP& problem, int number_of_steps, int num_evals_per_step);

long long get_number_of_operations(const Rosenbrock& problem, int number_of_steps, int num_evals_per_step);

// Gets the date in a nice format without spaces
std::string get_date();

// Saves the string to a file.
std::string saveStringToFile(std::string s, std::string subfolder, std::string file_suffix, std::string file_extension=".txt");

std::string getResultsString(const std::vector<float> scores, double time, double num_ops, std::string description);
#endif // __METRICS_H__