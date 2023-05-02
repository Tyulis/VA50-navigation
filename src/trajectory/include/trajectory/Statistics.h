#ifndef _TRAJECTORY_STATISTICS_H
#define _TRAJECTORY_STATISTICS_H

#include <cmath>
#include <vector>
#include <armadillo>

#include "trajectory/Utility.h"


/** Compute the confidence about some data inferred from a number of data points with their own confidence levels
 *  This computes 1 - [√[∑(1-c)²] / N]
 *  - confidences : std::vector<float> : Individual confidence scores, in [0, 1] */
inline float confidence_combination(std::vector<float> const& confidences) {
	float sq_inverted_sum = 0.0f;
	for (auto it = confidences.cbegin(); it != confidences.cend(); it++)
		sq_inverted_sum += sq(1 - *it);
	return 1 - std::sqrt(sq_inverted_sum) / confidences.size();
}

inline float confidence_combination(arma::fvec const& confidences) {
	float sq_inverted_sum = arma::sum(arma::square(1 - confidences));
	return 1 - std::sqrt(sq_inverted_sum) / confidences.n_elem;
}

inline float confidence_combination(arma::frowvec const& confidences) {
	float sq_inverted_sum = arma::sum(arma::square(1 - confidences));
	return 1 - std::sqrt(sq_inverted_sum) / confidences.n_elem;
}

#endif