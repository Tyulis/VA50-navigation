#ifndef _TEST_TESTREF_FUZZY_PARAMETERS
#define _TEST_TESTREF_FUZZY_PARAMETERS
#include <tuple>
#include <armadillo>

constexpr float _lane_width = 3.5;
constexpr float _main_angle_distance = 7.4;
constexpr float _vertical_angle_tolerance = 0.5;
std::tuple<arma::fvec, arma::fvec, arma::fvec, arma::fvec, arma::fmat, arma::fmat> get_testref_fuzzy_parameters();

#endif