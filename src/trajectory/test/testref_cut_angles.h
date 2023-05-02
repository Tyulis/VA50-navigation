#ifndef _TEST_TESTREF_CUT_ANGLES
#define _TEST_TESTREF_CUT_ANGLES
#include <vector>
#include <armadillo>

constexpr int _filter_size = 9;
constexpr float _filter_deviation = 1;
constexpr float _min_branch_length = 8;
constexpr int _min_branch_size = 2;
constexpr float _max_curvature = 0.034;
std::vector<std::vector<arma::fmat>> get_testref_cut_angles();

#endif