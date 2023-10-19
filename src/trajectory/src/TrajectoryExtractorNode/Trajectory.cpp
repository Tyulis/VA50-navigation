#include <algorithm>

#include "trajectory/Utility.h"
#include "trajectory/TrajectoryExtractorNode.h"


/** Add a new local trajectory to the history buffer, cleaning up the old ones as needed
  * - trajectory : DiscreteCurve : New trajectory, with scores and timestamp, to add to the buffer */
void TrajectoryExtractorNode::add_local_trajectory(DiscreteCurve const& trajectory) {
	m_trajectory_history.push_back(trajectory);
	m_local_trajectory_history.push_back(trajectory);
	
	if (m_trajectory_history.size() > config::trajectory::history_size)
		m_trajectory_history.erase(m_trajectory_history.begin(), m_trajectory_history.begin() + (m_trajectory_history.size() - config::trajectory::history_size));
	
	// Make the transform service discard velocity data older than what will be useful from this point on
	double min_timestamp = m_trajectory_history[0].timestamp.toSec();
	for (auto it = m_intersection_hints.begin(); it != m_intersection_hints.end(); it++) {
		if (it->timestamps[0].toSec() < min_timestamp)
			min_timestamp = it->timestamps[0].toSec();
	}
	drop_velocity(ros::Time(min_timestamp));
}


/** Estimate a trajectory from the lane markings extracted from a single frame, and add it to the trajectory history buffer
  * - timestamp     : ros::Time               : Timestamp of the received image
  * - left_line     : DiscreteCurve           : Left line as a discrete curve, with scores. May be invalid.
  * - right_line    : DiscreteCurve           : Right line as a discrete curve, with scores. May be invalid.
  * - viz           : cv::Mat[y, x, CV_8UC3]& : Visualisation image */
void TrajectoryExtractorNode::compile_trajectory(ros::Time timestamp, DiscreteCurve& left_line, DiscreteCurve right_line, cv::Mat& viz) {
	DiscreteCurve left_estimate, right_estimate;

	// Compute the trajectory estimate according to the left line
	if (left_line.is_valid()) {
		left_estimate = left_line;
		left_estimate.dilate(config::environment::lane_width / 2, -1);
		left_estimate.trim_angles(arma::datum::pi / 3);

		// Visualize the original line and the estimation
		if (config::node::visualize) {
			std::vector<cv::Point> viz_points_line = arma::conv_to<std::vector<cv::Point>>::from(target_to_birdeye_config(left_line.curve));
			std::vector<cv::Point> viz_points_estimate = arma::conv_to<std::vector<cv::Point>>::from(target_to_birdeye_config(left_estimate.curve));
			for (int i = 0; i < left_line.size(); i++)
				cv::drawMarker(viz, viz_points_line[i], cv::Scalar(0, 255, int(left_line.scores(i) * 255)), cv::MARKER_CROSS, 4);  // Red
			for (int i = 0; i < left_estimate.size(); i++)
				cv::drawMarker(viz, viz_points_estimate[i], cv::Scalar(12, 255, int(left_estimate.scores(i) * 255)), cv::MARKER_CROSS, 4);  // Orange
		}
	}

	// Compute the trajectory estimate according to the right line
	if (right_line.is_valid()) {
		right_estimate = right_line;
		right_estimate.dilate(config::environment::lane_width / 2, 1);
		right_estimate.trim_angles(arma::datum::pi / 3);

		// Visualize the original line and the estimation
		if (config::node::visualize) {
			std::vector<cv::Point> viz_points_line = arma::conv_to<std::vector<cv::Point>>::from(target_to_birdeye_config(right_line.curve));
			std::vector<cv::Point> viz_points_estimate = arma::conv_to<std::vector<cv::Point>>::from(target_to_birdeye_config(right_estimate.curve));
			for (int i = 0; i < right_line.size(); i++)
				cv::drawMarker(viz, viz_points_line[i], cv::Scalar(90, 255, int(right_line.scores(i) * 255)), cv::MARKER_CROSS, 4);  // Cyan
			for (int i = 0; i < right_estimate.size(); i++)
				cv::drawMarker(viz, viz_points_estimate[i], cv::Scalar(125, 255, int(right_estimate.scores(i) * 255)), cv::MARKER_CROSS, 4);  // Blue
		}
	}

	// If only one estimate is valid, use it directly, otherwise compile the trajectory from both estimates
	DiscreteCurve trajectory;
	if (left_estimate.is_valid()) {
		if (right_estimate.is_valid()) {
			arma::fvec start_point = {0, config::trajectory::trajectory_start};
			std::vector<DiscreteCurve> estimates = {left_estimate, right_estimate};
			trajectory = compile_line(estimates, config::trajectory::trajectory_score_threshold, start_point, config::trajectory::trajectory_step);
		} else {
			trajectory = left_estimate;
		}
	} else if (right_estimate.is_valid()) {
		trajectory = right_estimate;
	}

	if (!trajectory.is_valid() || trajectory.size() <= 3) {
		ROS_ERROR("No valid local trajectory estimate, fail");
		return;
	}

	// If the trajectory estimate is valid, smooth it and add it to the history buffer
	trajectory.timestamp = timestamp;
	trajectory.savgol_filter(7);

	if (m_current_trajectory.is_valid() && !m_navigation_mode.is_intersection()) {
		auto [transform, distance] = get_map_transforms(m_current_trajectory.timestamp, timestamp);
		DiscreteCurve local_current_trajectory = transform_positions(transform, m_current_trajectory);
		local_current_trajectory.curve.shed_cols(arma::find(local_current_trajectory.curve.row(1) <= config::birdeye::roi_y));

		if (local_current_trajectory.size() > 4) {
			// If the new trajectory is too far away from the existing one, invalidate it
			float parallel_distance = mean_parallel_distance(trajectory, local_current_trajectory);
			if (parallel_distance > config::trajectory::max_parallel_distance) {
				if (config::node::visualize) {
					std::vector<cv::Point> viz_points = arma::conv_to<std::vector<cv::Point>>::from(target_to_birdeye_config(trajectory.curve));
					cv::polylines(viz, viz_points, false, cv::Scalar(0, 255, 255), 2);  // Red
				}
				return;
			}
		}
	}

	add_local_trajectory(trajectory);

	if (config::node::visualize) {
		std::vector<cv::Point> viz_points = arma::conv_to<std::vector<cv::Point>>::from(target_to_birdeye_config(trajectory.curve));
		cv::polylines(viz, viz_points, false, cv::Scalar(30, 255, 255), 2);  // Yellow
	}
}


/** Pull the entire trajectory history buffer to the local frame of the given timestamp 
  * - timestamp : ros::Time                  : Timestamp at which to pull the trajectory history 
  * <------------ std::vector<DiscreteCurve> : Transformed trajectories */
std::vector<DiscreteCurve> TrajectoryExtractorNode::pull_trajectories(ros::Time timestamp) {
	if (m_trajectory_history.size() == 0)
		return std::vector<DiscreteCurve>();

	std::vector<ros::Time> trajectory_timestamps;
	for (auto it = m_trajectory_history.begin() ; it != m_trajectory_history.end(); it++)
		trajectory_timestamps.push_back(it->timestamp);
	
	auto [transforms, distances] = get_map_transforms(trajectory_timestamps, timestamp);
	return transform_positions(transforms, m_trajectory_history);
}

float TrajectoryExtractorNode::turn_radius(ros::Time timestamp, float intersection_distance) {
	// Get the previously estimated trajectories,
	// keep only the parts that are within the intersection,
	// then get their mean curvature, that is used afterwards to estimate the turn curvature

	// Take the mean of the curvatures
	arma::fvec curvatures(m_local_trajectory_history.size());
	for (int i = 0; i < m_local_trajectory_history.size(); i++) {
		DiscreteCurve cut_curve(m_local_trajectory_history[i].curve.cols(arma::find(m_local_trajectory_history[i].curve.row(1) > intersection_distance)));
		cut_curve.trim_angles(arma::datum::pi / 4);
		curvatures(i) = mean_curvature(cut_curve.curve);
	}

	// The curvature is in rad/m, take its inverse to get the curvature radius
	float marking_curvature = arma::mean(curvatures);

	// If it’s obviously wrong, clip it to default values
	// This won’t do much good, but well, let’s see where it goes and if it’s really awful we’ll make it panic
	if (marking_curvature == 0) {
		ROS_WARN("No curvature information found, fall back to the default turn radius");
		return config::intersection::default_turn_radius;
	} else {
		float trajectory_radius = 1 / marking_curvature;
		if (trajectory_radius < config::intersection::min_turn_radius) {
			ROS_WARN("Found turn radius of %.3f m deemed too sharp, clamp to %.3f m", trajectory_radius, config::intersection::min_turn_radius);
			return config::intersection::min_turn_radius;
		} else if (trajectory_radius > config::intersection::max_turn_radius) {
			ROS_WARN("Found turn radius of %.3f m deemed too wide, clamp to %.3f m", trajectory_radius, config::intersection::max_turn_radius);
			return config::intersection::max_turn_radius;
		} else {
			return trajectory_radius;
		}
	}
}


/** Precompute the trajectory to follow to go forward through an intersection
  * - image_timestamp       : ros::Time : Timestamp at which to estimate the trajectory
  * - intersection_distance : float     : Distance remaining until the intersection (most likely negative, the vehicle is already on the intersection) */
void TrajectoryExtractorNode::build_intersection_forward_trajectory(ros::Time image_timestamp, float intersection_distance) {
	m_rejoin_distance = config::intersection::default_rejoin_distance;

	// Cut the currently recorded trajectories at the intersection
	std::vector<DiscreteCurve> trajectories;
	for (auto it = m_local_trajectory_history.begin(); it != m_local_trajectory_history.end(); it++) {
		arma::uvec filter = arma::find(it->curve.row(1) < intersection_distance);
		trajectories.emplace_back(it->curve.cols(filter));
	}

	// Find the angle of the trajectory to generate
	float main_angle = estimate_main_angle(trajectories);

	// Just output a straight trajectory for as long as necessary
	arma::frowvec distances = arma::regspace<arma::frowvec>(0, config::trajectory::trajectory_step, 100);
	arma::fmat curve(2, distances.n_elem);
	curve.row(0) = distances * std::cos(main_angle);
	curve.row(1) = distances * std::sin(main_angle);

	DiscreteCurve trajectory(std::move(curve), image_timestamp);
	m_current_trajectory = trajectory;
}

/** Precompute the trajectory to follow to turn left at an intersection
  * - image_timestamp       : ros::Time : Timestamp at which to estimate the trajectory
  * - intersection_distance : float     : Distance remaining until the intersection (most likely negative, the vehicle is already on the intersection) */
void TrajectoryExtractorNode::build_intersection_left_trajectory(ros::Time image_timestamp, float intersection_distance) {
	// Basically, this is the same as a right turn, 
	// except the final trajectory is one radius further to accomodate the additonal lane to skip
	float trajectory_radius = turn_radius(image_timestamp, intersection_distance);	

	// Compute the trajectory as a little straight path then a quarter circle to the right with that radius
	float angle_step = config::trajectory::trajectory_step / trajectory_radius;
	arma::frowvec angles = arma::regspace<arma::frowvec>(0, angle_step, arma::datum::pi / 2);
	
	float init_distance = ((m_next_double_lane)? 1.75f : 1.0f) * config::environment::lane_width;
	int init_steps = std::floor(init_distance / config::trajectory::trajectory_step);

	arma::fmat curve(2, angles.n_elem + init_steps);
	if (init_steps > 0) {
		curve.submat(0, 0, 0, init_steps - 1).fill(0.0f);
		curve.submat(1, 0, 1, init_steps - 1) = arma::regspace<arma::fmat>(0, init_distance, config::trajectory::trajectory_step);
	}

	curve.submat(0, init_steps, 0, curve.n_cols - 1) = trajectory_radius * (arma::cos(angles) - 1);
	curve.submat(1, init_steps, 1, curve.n_cols - 1) = trajectory_radius * arma::sin(angles) + intersection_distance + init_distance;
	
	m_current_trajectory = DiscreteCurve(std::move(curve), image_timestamp);
	// m_rejoin_distance = std::max(config::environment::lane_width + intersection_distance + trajectory_radius * (config::intersection::rejoin_factor * std::sqrt(2.0f) / 2), config::intersection::default_rejoin_distance);
	m_rejoin_distance = -1;
}

/** Precompute the trajectory to follow to turn right at an intersection
  * - image_timestamp       : ros::Time : Timestamp at which to estimate the trajectory
  * - intersection_distance : float     : Distance remaining until the intersection (most likely negative, the vehicle is already on the intersection) */
void TrajectoryExtractorNode::build_intersection_right_trajectory(ros::Time image_timestamp, float intersection_distance) {
	float trajectory_radius = turn_radius(image_timestamp, intersection_distance);	
	ROS_INFO("Turn radius : %.3f", trajectory_radius);

	// Compute the trajectory as a quarter circle to the right with that radius
	float angle_step = config::trajectory::trajectory_step / trajectory_radius;
	arma::frowvec angles = arma::fliplr(arma::regspace<arma::frowvec>(arma::datum::pi / 2, angle_step, arma::datum::pi));
	arma::fmat curve(2, angles.n_elem);
	curve.row(0) = trajectory_radius * (1 + arma::cos(angles));
	curve.row(1) = trajectory_radius * arma::sin(angles) + intersection_distance;
	
	m_current_trajectory = DiscreteCurve(std::move(curve), image_timestamp);
	// m_rejoin_distance = std::max(intersection_distance + trajectory_radius * (config::intersection::rejoin_factor * std::sqrt(2.0f) / 2), config::intersection::default_rejoin_distance);
	m_rejoin_distance = -1;  // Delegate this to distance_until_rejoin, that compares the angle instead of distance. FIXME : Architecture-wise, this is crap.
}

/** Update the current trajectory to the given timestamp, using the trajectory history buffers
  * - target_timestamp : ros::Time : Timestamp for which the final trajectory must be estimated */
void TrajectoryExtractorNode::update_trajectory(ros::Time timestamp, cv::Mat& viz) {
	// Visualization of per-frame trajectories
	if (config::node::visualize) {
		for (auto it = m_local_trajectory_history.begin(); it != m_local_trajectory_history.end(); it++) {
			std::vector<cv::Point> viz_points = arma::conv_to<std::vector<cv::Point>>::from(target_to_birdeye_config(it->curve));
			for (int i = 0; i < viz_points.size(); i++)
				cv::drawMarker(viz, viz_points[i], cv::Scalar(150, 255, int(it->scores(i) * 255)), cv::MARKER_CROSS, 4);  // Purple
		}
	}

	// Compile a global trajectory from the local per-frame ones
	arma::fvec start_point = {0, config::trajectory::trajectory_start};
	DiscreteCurve compiled_trajectory = compile_line(m_local_trajectory_history, config::trajectory::trajectory_score_threshold, start_point, config::trajectory::trajectory_step);

	// Failure : make the current trajectory invalid
	if (!compiled_trajectory.is_valid() || compiled_trajectory.size() <= 3) {
		m_current_trajectory = DiscreteCurve();
		return;
	}

	compiled_trajectory.trim_angles(arma::datum::pi / 2);
	compiled_trajectory.savgol_filter(7);
	compiled_trajectory.timestamp = timestamp;

	// Cut the first points if the angles they make with the forward vector is too sharp
	while (compiled_trajectory.curve.n_cols > 0 && std::abs(M_PI / 2 - std::atan2(compiled_trajectory.curve(1, 0), compiled_trajectory.curve(0, 0))) > config::trajectory::max_output_angle)
		compiled_trajectory.curve.shed_col(0);
	
	if (compiled_trajectory.size() <= 3) {
		ROS_ERROR("Not enough points remaining after max output angle filter (%d points)", compiled_trajectory.size());
		m_current_trajectory = DiscreteCurve();
		return;
	}

	m_current_trajectory = compiled_trajectory;

	if (config::node::visualize) {
		std::vector<cv::Point> viz_points = arma::conv_to<std::vector<cv::Point>>::from(target_to_birdeye_config(compiled_trajectory.curve));
		cv::polylines(viz, viz_points, false, cv::Scalar(60, 255, 255), 2);  // Green
	}
}
