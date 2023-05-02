#ifndef _TEST_TESTREF_SAVGOL
#define _TEST_TESTREF_SAVGOL
#include <vector>
#include <armadillo>

constexpr float _savgol_window = 7;
constexpr float _savgol_degree = 2;
std::vector<arma::fmat> get_testref_savgol();

#endif