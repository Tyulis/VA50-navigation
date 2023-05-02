#ifndef _TEST_TESTREF_PROJECT_POINT
#define _TEST_TESTREF_PROJECT_POINT
#include <tuple>
#include <vector>
#include <armadillo>

std::tuple<arma::fmat, arma::fmat> get_testref_projection_curves();
std::vector<std::tuple<int, float>> get_testref_projection_results();

#endif