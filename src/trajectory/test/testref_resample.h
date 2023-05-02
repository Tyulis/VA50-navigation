#ifndef _TEST_TESTREF_RESAMPLE
#define _TEST_TESTREF_RESAMPLE
#include <vector>
#include <armadillo>

constexpr float _resample_step = 10;
std::vector<arma::fmat> get_testref_resample();

#endif