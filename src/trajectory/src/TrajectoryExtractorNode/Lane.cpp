#include <cmath>

#include "trajectory/NavigationMode.h"
#include "trajectory/DiscreteGeometry.h"
#include "trajectory/TrajectoryExtractorNode.h"


//                         ╔══════════════════╗                          #
// ════════════════════════╣  LANE DETECTION  ╠═════════════════════════ #
//                         ╚══════════════════╝                          #


/** Find the best candidate for the current lane
  * - lines           : std::vector<DiscreteCurve>& : Extracted and filtered curves in pixel coordinates
  * - scale_factor    : float                       : Scale factor from pixel to metric
  * - image_timestamp : ros::Time                   : Timestamp at which the image was captured
  * - markings        : std::vector<Marking>        : List of previously detected markings
  * <------------------ DiscreteCurve               : Left line as a discrete curve in pixel coordinates
  * <------------------ DiscreteCurve               : Right line as a discrete curve in pixel coordinates (both can be defined or invalid at the same time) */
std::tuple<DiscreteCurve, DiscreteCurve> TrajectoryExtractorNode::detect_lane(std::vector<DiscreteCurve>& lines, float scale_factor, ros::Time image_timestamp, std::vector<Marking> const& markings) {
	// Estimate the main angle of those lines
	// If we are on a road, chances are, a lot of curve segments belong to lane markings
	// In the few meters ahead of the vehicle, those have most likely approximately all the same angle
	// Thus, to estimate the angle of the lane relative to the vehicle, to estimate the expected position of the current
	// lane markings better, we just have to take the angle where the angles of curve segments are most concentrated
	float main_angle = estimate_main_angle(lines);

	// Build the fuzzy logic base variables for each curve
	auto [forward_distance, left_line_distance, right_line_distance, line_lengths, parallel_distances, parallel_angles]
		= fuzzy_lane_parameters(lines, main_angle, (config::birdeye::roi_y + config::fuzzy_lines::local_area_y) / 2);
	
	int left_index, right_index;
	float left_score, right_score;
	
	// Cruise mode : Detect the current lane, and fall back to a single marking if no sufficiently correct full lane is found
	if (m_navigation_mode == NavigationMode::Cruise)
		std::tie(left_index, right_index, left_score, right_score) = detect_full_lane(forward_distance, left_line_distance, right_line_distance, line_lengths, parallel_distances, parallel_angles, true);
	
	// Intersection modes, AFTER the rejoin distance has been reached :
	// The vehicle must now catch the next lane to follow, so to maximize the chances of getting it right,
	// force it to use only FULL lanes (left and right), or fail and wait for the next image 
	else if (m_navigation_mode.is_intersection())
		std::tie(left_index, right_index, left_score, right_score) = detect_full_lane(forward_distance, left_line_distance, right_line_distance, line_lengths, parallel_distances, parallel_angles, false);
	
	if (config::node::visualize)
		m_visualization.update_line_detection(lines, left_index, right_index, markings, scale_factor);
	
	DiscreteCurve left_marking, right_marking;
	if (left_index >= 0) {
		left_marking = lines[left_index];
		init_lane_scores(left_marking, left_score);
	}

	if (right_index >= 0) {
		right_marking = lines[right_index];
		init_lane_scores(right_marking, right_score);
	}

	return {left_marking, right_marking};
}

/**Detect a full lane (left and right marking) among a list of discrete curves, given some statistics about them
  * - forward_distance    : arma::fvec[N]    : Distance of the first valid point of the curve to the vehicle. The closer the better
  * - left_line_distance  : arma::fvec[N]    : Distance of the first valid point to the expected left marking. The closer the better
  * - right_line_distance : arma::fvec[N]    : Distance of the first valid point to the expected right marking. The closer the better
  * - line_lengths        : arma::fvec[N]    : Lengths of the curve from the first valid point. The longer the better
  * - parallel_distance   : arma::fmat[N, N] : Difference of the mean orthogonal distance between both curves to the expected lane width, in terms of lane widths, for each pair of curve. The lower the better
  * - parallel_angles     : arma::fmat[N, N] : Mean angle between the vectors of both curves, for each pair of curves, in radians. The lower the better
  * - fallback            : bool             : If true, when no sufficiently high-quality lane is found, fall back to finding only one of both markings instead.
  * <---------------------- int              : Index of the best left marking found among the initial list of curves, or -1
  * <---------------------- int              : Index of the best right marking found among the initial list of curves, or -1 (both can be defined or invalid at the same time)
  * <---------------------- float            : Reliability score of the left line in [0, 1], or -1
  * <---------------------- float            : Reliability score of the right line in [0, 1], or -1 */
std::tuple<int, int, float, float> TrajectoryExtractorNode::detect_full_lane(arma::fvec const& forward_distance, arma::fvec const& left_line_distance, arma::fvec const& right_line_distance,
				arma::fvec const& line_lengths, arma::fmat const& parallel_distances, arma::fmat const& parallel_angles, bool fallback) {
	// Put the variables in a [N, N, 5] cube for the fuzzy logic module
	// We need 2D variables, for each pair of curves, so we need to combine their 1D values
	// Forward distance and line length use the usual arithmetic mean, the distance to the expected markings
	// take the best combination of left and right among the pair, and takes the maximum distance among the combination
	int num_curves = forward_distance.n_elem;
	auto [FD_columns, FD_rows] = meshgrid(forward_distance);
	auto [LD_columns, LD_rows] = meshgrid(left_line_distance);
	auto [RD_columns, RD_rows] = meshgrid(right_line_distance);
	auto [LL_columns, LL_rows] = meshgrid(line_lengths);
	
	arma::fcube variables(num_curves, num_curves, 5);
	variables.slice(0) = (FD_columns + FD_rows) / 2;
	variables.slice(1) = arma::min(arma::max(LD_columns, RD_rows), arma::max(RD_columns, LD_rows));
	variables.slice(2) = (LL_columns + LL_rows) / 2;
	variables.slice(3) = parallel_distances;
	variables.slice(4) = parallel_angles;

	// Get the best combination with its score from the fuzzy system
	auto [best_y, best_x, best_score] = m_lane_fuzzysystem.fuzzy_best(variables);
	ROS_INFO("Best lane score %f for combination [%d, %d]", best_score, best_y, best_x);

	// No good enough lane detected : fall back to single lines if so parameterized
	if (best_score < config::fuzzy_lines::lane_selection_threshold) {
		if (fallback) {
			ROS_WARN("No viable lane detected : resorting to single lines");
			return detect_any_line(forward_distance, left_line_distance, right_line_distance, line_lengths);
		} else {
			ROS_ERROR("No viable lane detected, fail");
			return {-1, -1, 0, 0};
		}
	}

	// Take the best configuration of left and right line for the combination found,
	// This is made overly complicated because we want to get the best *individual* assignment
	// to each line, not just some least-square solution, because sometimes the full lane score is okay
	// but one line is spot-on while the other is trash
	//
	// For example, when it detects the adjacent lane, it has found the left marking of the current lane,
	// and the left marking of the other lane, that is completely wrong for the current lane
	//
	// In that case, a least-squares solution would assign the good marking to the right, and the bad to the left,
	// whereas we want to assign the good marking to the left of the current lane, where it belongs, and ditch
	// the bad marking afterwards
	int left_index, right_index;
	if (right_line_distance(best_y) < right_line_distance(best_x)) {
		// Y better than X for both lines : Y gets the closest one
		if (left_line_distance(best_y) < left_line_distance(best_x)) {
			if (left_line_distance(best_y) < right_line_distance(best_y)) {
				left_index = best_y;   right_index = best_x;
			} else {
				left_index = best_x;   right_index = best_y;
			}
		} else {
			left_index = best_x;   right_index = best_y;
		}
	} else {
		// X better than Y for both lines : X gets the closest one
		if (left_line_distance(best_x) < left_line_distance(best_y)) {
			if (left_line_distance(best_x) < right_line_distance(best_x)) {
				left_index = best_x;   right_index = best_y;
			} else {
				left_index = best_y;   right_index = best_x;
			}
		} else {
			left_index = best_y;   right_index = best_x;
		}
	}

	// Use the fuzzy system for single markings to estimate the individual confidence scores of the selected markings
	arma::fcube line_variables(1, 2, 5);
	line_variables(0, 0, 0) = forward_distance(left_index);                line_variables(0, 1, 0) = forward_distance(right_index);
	line_variables(0, 0, 1) = left_line_distance(left_index);              line_variables(0, 1, 1) = right_line_distance(right_index);
	line_variables(0, 0, 2) = line_lengths(left_index);                    line_variables(0, 1, 2) = line_lengths(right_index);
	line_variables(0, 0, 3) = parallel_distances(left_index, right_index); line_variables(0, 1, 3) = parallel_distances(left_index, right_index);
	line_variables(0, 0, 4) = parallel_angles(left_index, right_index);    line_variables(0, 1, 4) = parallel_angles(left_index, right_index);

	arma::fmat line_scores = m_lane_fuzzysystem.fuzzy_scores(line_variables);
	float left_score = line_scores(0, 0);
	float right_score = line_scores(0, 1);

	ROS_INFO("Respective line scores : %f -> [%f, %f]", best_score, left_score, right_score);

	// Sometimes the full lane score is just okayish, because one of the curves is spot-on but the other is blatantly wrong
	// In that case, discard the bad one and continue only with the good one
	if (left_score < config::fuzzy_lines::single_line_selection_threshold) {
		ROS_WARN("Found a full lane but the left line’s score is too low, discarding the left line");
		return {-1, right_index, 0, right_score};
	} else if (right_score < config::fuzzy_lines::single_line_selection_threshold) {
		ROS_WARN("Found a full lane but the right line’s score is too low, discarding the right line");
		return {left_index, -1, left_score, 0};
	} else {
		return {left_index, right_index, left_score, right_score};
	}
}


/** Detect the best single marking to constrain the current lane
  * - forward_distance : arma::fvec[N] : Distance of the first valid point of the curve to the vehicle. The closer the better
  * - line_distance    : arma::fvec[N] : Distance of the first valid point to the expected marking. The closer the better
  * - line_lengths     : arma::fvec[N] : Lengths of the curve from the first valid point. The longer the better
  * <------------------- int, int      : Index of the best marking found among the initial list of curves, or -1.
                                         One of those is -1, depending on the best side for the detected marking
  * <------------------- float, float  : Reliability score of the line in [0, 1]. Same thing about the order */
std::tuple<int, int, float, float> TrajectoryExtractorNode::detect_any_line(arma::fvec const& forward_distance, arma::fvec const& left_line_distance, arma::fvec const& right_line_distance, arma::fvec const& line_lengths) {
	// Emulate the 2D variables with the best value, such that it has no impact (malus 0)
	arma::fvec line_distance = arma::min(left_line_distance, right_line_distance);

	int num_curves = forward_distance.n_elem;
	arma::fcube variables(num_curves, 1, 5);
	variables.slice(0) = forward_distance;
	variables.slice(1) = line_distance;
	variables.slice(2) = line_lengths;
	variables.slice(3) = arma::ones<arma::fmat>(num_curves, 1) * config::fuzzy_lines::centers::parallel_distances[0];
	variables.slice(4) = arma::ones<arma::fmat>(num_curves, 1) * config::fuzzy_lines::centers::parallel_angles[0];

	auto [best_index, best_x, best_score] = m_lane_fuzzysystem.fuzzy_best(variables);
	ROS_INFO("Best single line score : %f", best_score);
	if (best_score < config::fuzzy_lines::single_line_selection_threshold) {
		ROS_ERROR("No viable single line detected, fail");
		return {-1, -1, 0, 0};
	}

	if (left_line_distance(best_index) < right_line_distance(best_index)) {
		return {best_index, -1, best_score, 0};
	} else {
		return {-1, best_index, 0, best_score};
	}
}

/** Initialize the confidence score of each point of the curve, given a score for the whole curve
  * - line       : DiscreteCurve& : Input marking, the scores are filled directly within it
  * - base_score : float          : Score for the entire `line` */
void TrajectoryExtractorNode::init_lane_scores(DiscreteCurve& line, float base_score) {
	arma::frowvec distances = arma::cumsum(column_norm(arma::diff(line.curve, 1, 1)));
	arma::frowvec factors = 1 - arma::exp((distances - config::trajectory::line_reliability_range) / config::trajectory::line_reliability_dampening);
	line.scores = arma::frowvec(line.curve.n_cols);
	line.scores(0) = base_score;
	line.scores(arma::span(1, line.curve.n_cols - 1)) = base_score * factors.clamp(0, 1);
}
