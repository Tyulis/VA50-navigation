#ifndef _TEST_TESTREF_DILATE
#define _TEST_TESTREF_DILATE

#include <vector> 
#include <armadillo>

constexpr float _dilation = 1.75;
constexpr int _left_direction = -1;
constexpr int _right_direction = 1;
std::vector<arma::fmat> get_testref_dilate_left();
std::vector<arma::fmat> get_testref_dilate_right();

#endif