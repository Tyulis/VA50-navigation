#ifndef _TRAJECTORY_FUZZYLANESYSTEM_H
#define _TRAJECTORY_FUZZYLANESYSTEM_H

#include <tuple>
#include <armadillo>

#include "trajectory/DiscreteCurve.h"


class FuzzyLaneSystem {
	public:
		FuzzyLaneSystem() = default;
		FuzzyLaneSystem(arma::fmat const& centers, arma::imat const& malus, arma::fvec const& output_centers, int base_score);

		std::tuple<int, int, float> fuzzy_best(arma::fcube const& variables) const;
		arma::fmat fuzzy_scores(arma::fcube const& variables) const;

	private:
		float cell_score(arma::fvec const& variables) const;
		arma::fvec cell_conditions(arma::fvec const& variables) const;
		arma::fmat fuzzify(arma::fvec const& variables) const;

		int m_num_variables;
		int m_num_subsets;
		int m_num_outputs;
		int m_num_rules;

		arma::ivec m_ruleset;
		arma::uvec m_rulegroup_counts;
		arma::fmat m_centers;
		arma::fmat m_output_centers;
};

float estimate_main_angle(std::vector<DiscreteCurve> const& lines);
std::tuple<arma::fvec, arma::fvec, arma::fvec, arma::fvec, arma::fmat, arma::fmat> fuzzy_lane_parameters(std::vector<DiscreteCurve> const& lines, float main_angle, float main_angle_distance);

#endif