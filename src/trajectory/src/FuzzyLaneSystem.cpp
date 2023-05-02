#include <cmath>
#include <cassert>
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde_model.hpp>

#include "config.h"
#include "trajectory/FuzzyLaneSystem.h"



/** Initialize the fuzzy system and precompute the ruleset
  * V = number of variables, S = number of fuzzy subsets per variable, C = number of output sets
  * - centers        : arma::fmat[V, S] : Centers for each set of each variable
  * - malus          : arma::imat[V, S] : Malus applied to the base score for each set of each variable
  * - output_centers : arma::fvec[C]    : Center for each set of the output variable
  * - base_score     : int              : Base score for the ruleset, translates to output variables sets after malus */
FuzzyLaneSystem::FuzzyLaneSystem(arma::fmat const& centers, arma::imat const& malus, arma::fvec const& output_centers, int base_score) {
	m_num_variables = centers.n_rows;
	m_num_subsets = centers.n_cols;
	m_num_outputs = output_centers.n_elem;
	m_num_rules = std::pow(m_num_subsets, m_num_variables);

	assert(m_num_subsets == 3);

	m_centers = centers;
	m_output_centers = output_centers;
	m_rulegroup_counts = arma::uvec(m_num_outputs, arma::fill::zeros);
	m_ruleset = arma::ivec(m_num_rules);
	
	// Precompute the ruleset
	// Each rule is identified by a unique index built from the sets of each variable that lead to it
	// For example, with 3 sets per variables, it would be something like 10122 in base 3 for sets 1, 0, 1, 2, and 2 of the respective variables
	// As such, it can accomodate any amount of variables within a single dimension, no need for cumbersome arbitrary dimensional arrays
	for (int rule_index = 0; rule_index < m_num_rules; rule_index++) {
		int score = base_score;

		// Decompose the index into the respective set indices for each variable,
		// and add the corresponding malus
		int temp_index = rule_index;
		for (int var_index = 0; var_index < m_num_variables; var_index++) {
			int set_index = temp_index % m_num_subsets;
			temp_index /= m_num_subsets;
			score += malus(var_index, set_index);
		}

		// Clamp to zero
		if (score < 0) score = 0;

		// Set the rule, and update the counts (number of rules that lead to each output subset)
		m_ruleset(rule_index) = score;
		m_rulegroup_counts(score) += 1;
	}
}


/** Retrieve the best among all given pairs
  * - variables : arma::fcube[V, y, x] : Value of each input variable for each element of the input grid
  * <------------ int                  : Y index (first index) of the best pair
  * <------------ int                  : X index (second index) of the best pair
  * <------------ float                : Score of the best pair */
std::tuple<int, int, float> FuzzyLaneSystem::fuzzy_best(arma::fcube const& variables) const {
	int best_y = -1, best_x = -1;
	float best_score = 0;

	// Infer with sum-product weighted heights inference and keep the pair with the highest score
	// Lower index is better
	for (int x = 0; x < variables.n_cols; x++) {
		for (int y = 0; y < variables.n_rows; y++) {
			arma::fvec cell_variables = variables.tube(y, x);
			float score = cell_score(cell_variables);

			// Keep only the max
			if (score > best_score) {
				best_score = score;
				best_y = y;
				best_x = x;
			}
		}
	}

	return {best_y, best_x, best_score};
}


arma::fmat FuzzyLaneSystem::fuzzy_scores(arma::fcube const& variables) const {
	arma::fmat scores = arma::zeros<arma::fmat>(variables.n_rows, variables.n_cols);

	// Infer with sum-product weighted heights inference and keep the pair with the highest score
	// Lower index is better
	for (int x = 0; x < variables.n_cols; x++) {
		for (int y = 0; y < variables.n_rows; y++) {
			arma::fvec cell_variables = variables.tube(y, x);
			scores(y, x) = cell_score(cell_variables);
		}
	}

	return scores;
}


/** Compute the score for a given input cell
  * - variables : arma::fvec[V] : Values of each variable V for some input
  * <------------ float         : Output score for that input cell */
float FuzzyLaneSystem::cell_score(arma::fvec const& variables) const {
	arma::fvec conditions = cell_conditions(variables);

	float numerator = arma::accu((conditions / m_rulegroup_counts) % m_output_centers);
	float denominator = arma::accu(conditions / m_rulegroup_counts);
	return numerator / denominator;
}


/** Compute the condition values for each output 
  * - variables : arma::fvec[V] : Values of each variable V for some input
  * <------------ arma::fvec[C] : Condition value of each output value for this point */
arma::fvec FuzzyLaneSystem::cell_conditions(arma::fvec const& variables) const {
	arma::fmat membership = fuzzify(variables);

	// We need to get the condition coefficient for each output center,
	// So start at 0 and calculate the mean over all rules in the ruleset for each pair
	// Same logic as before, the rules’ indices are their input fuzzy sets encoded in base 3
	arma::fvec conditions(m_num_outputs, arma::fill::zeros);
	for (int rule_index = 0; rule_index < m_num_rules; rule_index++) {
		float condition = 1;
		int temp_index = rule_index;
		for (int var_index = 0; var_index < m_num_variables; var_index++) {
			int set_index = temp_index % m_num_subsets;
			temp_index /= m_num_subsets;
			condition *= membership(var_index, set_index);
		}
		conditions(m_ruleset(rule_index)) += condition;
	}

	return conditions;
}



/** Fuzzify the given variables according to the set centers
  * - variables : arma::fvec[V]    : Values of each variable V for some input
  * <------------ arma::fmat[V, S] : Output, fuzzy membership coefficient of each variable V to each set S for that input */
arma::fmat FuzzyLaneSystem::fuzzify(arma::fvec const& variables) const {
	arma::fmat results(m_num_variables, m_num_subsets);
	for (int var_index = 0; var_index < m_num_variables; var_index++) {
		// Check whether the centers are reversed (higher is better instead of lower is better), as it flips the whole logic
		// The convention for the output of this function is best first, worst last, regardless of the input logic
		bool reverse = m_centers(var_index, 0) > m_centers(var_index, 1);
		
		if (!reverse) {
			// Before the first center ⟶ 100% good
			if (variables(var_index) < m_centers(var_index, 0)) {
				results(var_index, 0) = 1;
				results(var_index, 1) = 0;
				results(var_index, 2) = 0;
			}
			
			// Between the first and second centers ⟶ triangular good/medium
			else if (variables(var_index) < m_centers(var_index, 1)) {
				results(var_index, 1) = (variables(var_index) - m_centers(var_index, 0)) / (m_centers(var_index, 1) - m_centers(var_index, 0));
				results(var_index, 0) = 1 - results(var_index, 1);
				results(var_index, 2) = 0;
			}

			// Between the second and third centers ⟶ triangular medium/bad
			else if (variables(var_index) < m_centers(var_index, 2)) {
				results(var_index, 0) = 0;
				results(var_index, 2) = (variables(var_index) - m_centers(var_index, 1)) / (m_centers(var_index, 2) - m_centers(var_index, 1));
				results(var_index, 1) = 1 - results(var_index, 2);
			}

			// After the last center ⟶ 100% bad
			else {
				results(var_index, 0) = 0;
				results(var_index, 1) = 0;
				results(var_index, 2) = 1;
			}
		}

		// Reversed : higher is better, indices are reversed
		else {
			// After the last center ⟶ 100% bad
			if (variables(var_index) < m_centers(var_index, 2)) {
				results(var_index, 2) = 1;
				results(var_index, 1) = 0;
				results(var_index, 0) = 0;
			}
			
			// Between the third and second centers ⟶ triangular medium/bad
			else if (variables(var_index) < m_centers(var_index, 1)) {
				results(var_index, 1) = (variables(var_index) - m_centers(var_index, 2)) / (m_centers(var_index, 1) - m_centers(var_index, 2));
				results(var_index, 2) = 1 - results(var_index, 1);
				results(var_index, 0) = 0;
			}

			// Between the first and second centers ⟶ triangular good/medium
			else if (variables(var_index) < m_centers(var_index, 0)) {
				results(var_index, 2) = 0;
				results(var_index, 0) = (variables(var_index) - m_centers(var_index, 1)) / (m_centers(var_index, 0) - m_centers(var_index, 1));
				results(var_index, 1) = 1 - results(var_index, 0);
			}

			// Before the first center ⟶ 100% good
			else {
				results(var_index, 2) = 0;
				results(var_index, 1) = 0;
				results(var_index, 0) = 1;
			}
		}
	}

	return results;
}



/** Estimate the main angle of detected lines in some area ahead of the vehicle,
  * to better estimate the current lane’s expected position
  * - lines : std::vector<DiscreteCurve> : Discrete curves in the image
  * <-------- float                      : Principal angle of those lines, in radians */
float estimate_main_angle(std::vector<DiscreteCurve> const& lines) {
	// First, get the local angles between each curve segment
	// For some reason, the kernel density estimation only accepts doubles
	arma::dvec angles;
	for (auto it = lines.begin(); it != lines.end(); it++) {
		// Filter out the points that are not in the "local area" of estimation
		arma::uvec local_filter = arma::find((-config::fuzzy_lines::local_area_x < it->curve.row(0)) && (it->curve.row(0) < config::fuzzy_lines::local_area_x) &&
		                                     (0 < it->curve.row(1)) && (it->curve.row(1) < config::fuzzy_lines::local_area_y));
		
		// No angle remaining
		if (arma::sum(local_filter) < 2)
			continue;
		
		arma::fmat gradient = it->gradient().cols(local_filter);
		arma::frowvec line_angles = arma::atan2(gradient.row(1), gradient.row(0));
		angles.insert_rows(angles.n_elem, arma::conv_to<arma::dvec>::from(line_angles));
	}

	// No data to estimate from : assume straight ahead (π/2)
	if (angles.n_elem == 0)
		return M_PI / 2;
	
	// Remove the angles that are a bit too extreme
	angles.shed_rows(arma::find((config::fuzzy_lines::main_angle_cut >= angles) || (angles >= arma::datum::pi - config::fuzzy_lines::main_angle_cut)));
	if (angles.n_elem == 0)
		return M_PI / 2;
	
	// Return the maximum density estimation among the extracted angles
	mlpack::util::Timers timers;
	arma::dvec estimations;
	arma::dvec query_angles = angles;
	mlpack::KDEModel density_model(0.1, 0.005, 0, mlpack::KDEModel::KernelTypes::EPANECHNIKOV_KERNEL);
	density_model.BuildModel(timers, std::move(angles));
	density_model.Evaluate(timers, estimations);
	return float(query_angles(arma::index_max(estimations)));
}


/** Compute the parameters given to the fuzzy systems from the discrete curves
  * - lines               : std::vector<DiscreteCurve>[M] : Input discrete curves, in metric coordinates in the local road frame
  * - main_angle          : float                         : Expected angle of the lane relative to the vehicle, in radians
  * - main_angle_distance : float                         : Distance at which that angle is reached
  * <---------------------- arma::fvec[M]                 : Distance of the initial point of each curve to the vehicle
  * <---------------------- arma::fvec[M]                 : Orthogonal distance of each initial point to the expected left lane marking
  * <---------------------- arma::fvec[M]                 : Orthogonal distance of each initial point to the expected right lane marking
  * <---------------------- arma::fvec[M]                 : Length of each curve from their initial point
  * <---------------------- arma::fmat[M, M]              : Average distance between each pair of curves, in terms of expected lane widths
  * <---------------------- arma::fmat[M, M]              : Average angle between each pair of curves, in radians */
std::tuple<arma::fvec, arma::fvec, arma::fvec, arma::fvec, arma::fmat, arma::fmat> fuzzy_lane_parameters(std::vector<DiscreteCurve> const& lines, float main_angle, float main_angle_distance) {
	// Create the arrays
	arma::ivec initial_indices(lines.size());
	arma::fvec forward_distance(lines.size());
	arma::fvec left_line_distance(lines.size());
	arma::fvec right_line_distance(lines.size());
	arma::fvec line_lengths(lines.size());
	arma::fmat parallel_distances(lines.size(), lines.size());
	arma::fmat parallel_angles(lines.size(), lines.size());

	// Precompute the expected markings
	// This is a bit complicated because assuming they are at `main_angle` straight ahead would completely disregard
	// the lateral offset and rotation of the vehicle relative to the lane

	// So what we do here, is fitting a circle arc from the current vehicle position,
	// such that it reaches `main_angle` relative to the vehicle when y = main_angle_distance
	// This gives us a lateral offset for the expected lane center, that gets at [offset, main_angle_distance] instead of [0, main_angle_distance]
	// Then we get base points for the left and right markings by orthogonally offsetting this main point by a half lane width on each side 
	// From that, the expected markings are defined by those main points and the expected vector extracted from the main angle
	// This complicates things a bit, but given our far, far away visibility window, it is unavoidable
	arma::fvec expected_vector = {std::cos(main_angle), std::sin(main_angle)};
	arma::fvec orthogonal_vector = {-std::sin(main_angle), std::cos(main_angle)};
	arma::fvec main_point = {0, main_angle_distance};  // This value would assume a PI/2 main angle (straight ahead)
	if (main_angle < M_PI / 2)                         // Curving to the right
		main_point(0) = (main_angle_distance / std::cos(main_angle)) * (1 - std::sin(main_angle));
	else if (main_angle > M_PI / 2)                    // Curving to the left
		main_point(0) = (main_angle_distance / std::cos(main_angle)) * (std::sin(main_angle) - 1);
	
	arma::fvec left_main_point = main_point + orthogonal_vector * config::environment::lane_width/2;
	arma::fvec right_main_point = main_point - orthogonal_vector * config::environment::lane_width/2;

	// Now compute the parameters for each curve
	for (int i = 0; i < lines.size(); i++) {
		DiscreteCurve curve = lines[i];

		// First, get the "initial point"
		// This is the first point at which the relative angle between the curve and the vehicle gets
		// reasonably close to the main angle, to eliminate completely irrelevant curves right avay
		// and disregard weird starts for some curves
		arma::fmat vectors = arma::diff(curve.curve, 1, 1);
		arma::frowvec angles = arma::atan2(vectors.row(1), vectors.row(0));
		arma::uvec tolerated = arma::find((main_angle - config::fuzzy_lines::vertical_angle_tolerance < angles) &&
		                                  (angles < main_angle + config::fuzzy_lines::vertical_angle_tolerance));
		
		// The curve is completely out of line : make it definitely invalid for the fuzzy system
		if (tolerated.n_elem == 0) {
			initial_indices(i) = -1;
			forward_distance(i) = arma::datum::inf;
			left_line_distance(i) = arma::datum::inf;
			right_line_distance(i) = arma::datum::inf;
			line_lengths(i) = 0;
			continue;
		}

		// Compute the actual parameters
		int initial_index = tolerated(0);
		initial_indices(i) = initial_index;
		forward_distance(i) = curve.curve(1, initial_index);
		left_line_distance(i) = distance_point_to_line(curve.curve.col(initial_index), left_main_point, main_angle) / config::environment::lane_width;
		right_line_distance(i) = distance_point_to_line(curve.curve.col(initial_index), right_main_point, main_angle) / config::environment::lane_width;
		line_lengths(i) = curve_length(curve.curve.cols(initial_index, curve.size() - 1));
	}

	// The individual parameters are done, but we have yet to compute the pairwise parameters
	for (int i = 0; i < lines.size(); i++) {
		// A combination of a marking with itself is invalid
		parallel_distances(i, i) = arma::datum::inf;
		parallel_angles(i, i) = arma::datum::inf;

		for (int j = i + 1; j < lines.size(); j++) {
			// One marking is invalid ⟶ the pair is invalid
			if (initial_indices(i) < 0 || initial_indices(j) < 0) {
				parallel_distances(i, j) = parallel_distances(j, i) = arma::datum::inf;
				parallel_angles(i, j) = parallel_angles(j, i) = arma::datum::inf;
				continue;
			}

			DiscreteCurve longest_line, shortest_line;
			if (line_lengths(i) > line_lengths(j)) {
				longest_line = lines[i];
				shortest_line = lines[j];
			} else {
				longest_line = lines[j];
				shortest_line = lines[i];
			}

			// Compute the average orthogonal distance and the average angle difference
			int valid_points = 0;
			float paralleldiff = 0, anglediff = 0;
			arma::fmat vectors_long = arma::diff(longest_line.curve, 1, 1);  // longest_line.gradient();
			arma::fmat vectors_short = arma::diff(shortest_line.curve, 1, 1);  // shortest_line.gradient();
			for (int p = 0; p < longest_line.size() - 1; p++) {
				auto [index, segment_part] = shortest_line.project_point(longest_line.curve.col(p), false);
				if (index < 0)
					continue;

				float actual_distance = arma::norm(longest_line.get_point(p) - shortest_line.get_point(index, segment_part));
				paralleldiff += std::abs(actual_distance - config::environment::lane_width) / config::environment::lane_width;
				anglediff += std::abs(vector_angle(vectors_long.col(p), vectors_short.col(index)));
				valid_points += 1;
			}

			if (valid_points == 0) {
				parallel_distances(i, j) = parallel_distances(j, i) = arma::datum::inf;
				parallel_angles(i, j) = parallel_angles(j, i) = arma::datum::inf;
			} else {
				parallel_distances(i, j) = parallel_distances(j, i) = paralleldiff / valid_points;
				parallel_angles(i, j) = parallel_angles(j, i) = anglediff / valid_points;
			}
		}
	}

	return {forward_distance, left_line_distance, right_line_distance, line_lengths, parallel_distances, parallel_angles};
}