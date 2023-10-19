#include <mlpack/methods/dbscan.hpp>

#include "trajectory/Utility.h"
#include "trajectory/Statistics.h"
#include "trajectory/TrajectoryExtractorNode.h"


//                 ╔═══════════════════════════════════╗                 #
// ════════════════╣      INTERSECTION MANAGEMENT      ╠════════════════ #
//                 ╚═══════════════════════════════════╝                 #

/** Add a new intersection hint, that may be added to the list of hints,
 *  or merged with an existing hint if they are deemed different measurements of the same object
 *  - hint : IntersectionHint : New hint to consider */
void TrajectoryExtractorNode::add_intersection_hint(IntersectionHint const& hint) {
	// Skip hints while already in intersection mode
	if (m_navigation_mode != NavigationMode::Cruise)
		return;
	
	// Skip hints found too close to the last intersection
	if (!m_last_lane_rejoin.isZero()) {
		// Evaluate where the rejoined trajectory starts
		float min_start_distance = INFINITY;
		for (auto it = m_local_trajectory_history.begin(); it != m_local_trajectory_history.end(); it++) {
			float trajectory_start_distance = arma::norm(it->curve.col(0));
			if (it->curve(0, 1) < 0)
				trajectory_start_distance = -trajectory_start_distance;

			if (trajectory_start_distance < min_start_distance)
				min_start_distance = trajectory_start_distance;
		}

		auto [transform, distance] = get_map_transforms(m_last_lane_rejoin, hint.timestamps.back());
		
		// Too close to the previous intersection, skip
		if (distance < min_start_distance + config::intersection::hint_detection_buffer)
			return;
		// Now far enough, invalidate m_last_lane_rejoin to skip the transform service call next times
		else
			m_last_lane_rejoin = ros::Time(0);
	}

	// Check whether this hint has already been resolved before
	for (auto it = m_intersection_hints.begin(); it != m_intersection_hints.end(); it++) {
		if (match_intersection_hint(*it, hint)) {
			it->merge(hint);
			return;
		}
	}

	// No matching existing hint found, add to the list as a new one
	m_intersection_hints.push_back(hint);
}

/** Check whether two hints, an existing one and a new one, are different measurements of the same object 
 *  - existing_hint : IntersectionHint : Existing hint, evaluates about all its data
 *  - hint          : IntersectionHint : New hint, evaluates only its last data point (theoretically its only one)
 *  <---------------- bool             : True if the hints can be merged, false otherwise */
bool TrajectoryExtractorNode::match_intersection_hint(IntersectionHint const& existing_hint, IntersectionHint const& hint) {
	if (hint.category != existing_hint.category || hint.type != existing_hint.type)
		return false;
	
	// Take the centroid of the existing hint positions at the timestamp of the new hint measurement
	auto [transforms, distances] = get_map_transforms(existing_hint.timestamps, hint.timestamps.back());
	arma::fmat existing_positions = transform_positions(transforms, existing_hint.positions);
	arma::fvec existing_centroid = arma::mean(existing_positions, 1);
	return arma::norm(existing_centroid - hint.positions.back()) < IntersectionHint::match_threshold(hint.category);
}

/** Check for the distance to the next estimated intersection, and switch navigation mode if necessary
  * - image_timestamp : ros::Time : Timestamp to estimate the informations about */
void TrajectoryExtractorNode::update_intersection(ros::Time image_timestamp) {
	// Don’t estimate the next intersection if the vehicle is already on an intersection
	if (m_navigation_mode != NavigationMode::Cruise)
		return;

	auto [intersection_distance, intersection_directions] = next_intersection(image_timestamp);
	ROS_INFO("Next intersection : %f m, directions %s", intersection_distance, std::to_string(intersection_directions).c_str());

	// Close enough to the intersection, switch navigation mode
	if (intersection_directions != Direction::None && intersection_distance < config::intersection::mode_switch_distance) {
		// The chosen direction is not allowed at this intersection
		if (!(intersection_directions & m_next_direction)) {
			if (intersection_directions == Direction::None) {
				switch_panic(NavigationMode::PanicInvalid, image_timestamp, "No direction available at the next intersection");
				return;
			}
			// Only one direction available : follow it
			else if (m_next_direction.is_single_direction()) {
				m_next_direction = intersection_directions;
			}
			// Forward allowed : go forward and keep the current direction for next time
			else if (intersection_directions & Direction::Forward) {
				switch_intersection(NavigationMode::IntersectionForward, image_timestamp, intersection_distance);
				return;
			}
			// Only left or right, no direction chosen : panic, wait for input
			else {
				switch_panic(NavigationMode::PanicNoDirection, image_timestamp, "No direction chosen");
				return;
			}
		}

		// Switch to the relevant intersection mode
		else if (m_next_direction == Direction::Left)
			switch_intersection(NavigationMode::IntersectionLeft, image_timestamp, intersection_distance);
		else if (m_next_direction == Direction::Right)
			switch_intersection(NavigationMode::IntersectionRight, image_timestamp, intersection_distance);
		else if (m_next_direction == Direction::Forward)
			switch_intersection(NavigationMode::IntersectionForward, image_timestamp, intersection_distance);
		else
			switch_panic(NavigationMode::PanicInvalid, image_timestamp, "Invalid next direction " + std::to_string(m_next_direction));
	}
}

/** Estimate the position and directions of the next intersection 
 *  - image_timestamp : ros::Time : Timestamp to do the estimation at
 *  <------------------ float     : Distance to the next intersection, or -1 if no data available
 *  <------------------ Direction : Directions available at the next intersection, or Direction::None if no data available */
std::tuple<float, Direction> TrajectoryExtractorNode::next_intersection(ros::Time image_timestamp) {
	if (m_intersection_hints.empty())
		return {-1, Direction::None};
	
	std::vector<arma::fvec> hint_positions;
	std::vector<ros::Time> hint_timestamps;
	std::vector<int> hint_indices_vec;
	int index = 0;
	for (auto it = m_intersection_hints.begin(); it != m_intersection_hints.end(); it++) {
		hint_positions.insert(hint_positions.end(), it->positions.begin(), it->positions.end());
		hint_timestamps.insert(hint_timestamps.end(), it->timestamps.begin(), it->timestamps.end());
		hint_indices_vec.insert(hint_indices_vec.end(), it->positions.size(), index);
		index += 1;
	}
	arma::uvec hint_indices = arma::conv_to<arma::uvec>::from(hint_indices_vec);

	if (hint_positions.empty())
		return {-1, Direction::None};

	// Transform to the current timestamp
	auto [transforms, distances] = get_map_transforms(hint_timestamps, image_timestamp);
	arma::fmat transformed = transform_positions(transforms, hint_positions);

	// Project everything onto the current directional vector (0, 1, 0) and disregard points that are behind the vehicle
	arma::dmat current_distances = arma::conv_to<arma::dmat>::from(transformed.row(1));
	arma::uvec forward_filter = arma::find(current_distances >= config::intersection::hint_y_threshold);
	current_distances = current_distances(forward_filter);
	hint_indices = hint_indices(forward_filter);

	if (current_distances.is_empty())
		return {-1, Direction::None};

	// Less than two hints : just take the closest one
	if (current_distances.n_elem <= 2) {
		arma::uword selected_index = current_distances.index_min();
		IntersectionHint hint = m_intersection_hints[hint_indices(selected_index)];
		
		if (hint.confidence() < config::intersection::min_confidence)
			return {-1, Direction::None};
		
		return {current_distances(selected_index), hint.direction_hint()};
	}

	// Cluster the remaining hints with a loose DBSCAN
	arma::Row<size_t> assignments;
	arma::mat centroids;
	mlpack::DBSCAN dbscan(config::intersection::hint_clustering_distance, 2);
	dbscan.Cluster<arma::mat>(current_distances, assignments, centroids);

	// No meaningful clustering possible : again, just take the closest one
	if (arma::min(assignments) == SIZE_MAX) {
		arma::uword selected_index = current_distances.index_min();
		IntersectionHint hint = m_intersection_hints[hint_indices(selected_index)];
		
		if (hint.confidence() < config::intersection::min_confidence)
			return {-1, Direction::None};
		
		return {current_distances(selected_index), hint.direction_hint()};
	}

	// Take the cluster associated with the closest centroid
	arma::uword selected_cluster = centroids.index_min();
	float intersection_distance = centroids[selected_cluster];
	arma::uvec selected_positions = arma::find(assignments == selected_cluster);
	arma::uvec selected_indices = hint_indices(selected_positions);
	
	// For now, don’t keep the intersection if it is only given by traffic signs, too unreliable
	bool all_trafficsigns = true;
	selected_indices.for_each([&](arma::uvec::elem_type& index) {
		all_trafficsigns &= m_intersection_hints[index].category == IntersectionHint::Category::TrafficSign;
	});
	if (all_trafficsigns)
		return {-1, Direction::None};

	// Only keep the intersection if the combined hints give a high enough confidence
	std::vector<float> confidences;
	selected_indices.for_each([&](arma::uvec::elem_type& index) {
		confidences.insert(confidences.end(), m_intersection_hints[index].confidences.begin(), m_intersection_hints[index].confidences.end());
	});
	float confidence = confidence_combination(confidences);
	if (confidence < config::intersection::min_confidence)
		return {-1, Direction::None};

	Direction intersection_directions = Direction::All;
	arma::unique(selected_indices).eval().for_each([&] (int const& hint_index) {
		intersection_directions &= m_intersection_hints[hint_index].direction_hint();
	});

	return {intersection_distance, intersection_directions};
}

/** In intersection navigation modes, the vehicle follows a predetermined trajectory,
  * then tries to catch the new lane to follow after some distance stored in `m_rejoin_distance`
  * Check the distance remaining until that distance is reached
  * - image_timestamp : ros::Time : Timestamp to measure the distance at
  * <------------------ float      : Signed distance until the rejoin distance
		                             (negative if the vehicle is already farther than `m_rejoin_distance`) */
float TrajectoryExtractorNode::distance_until_rejoin(ros::Time image_timestamp) {
	auto [transform, distance] = get_map_transforms(m_last_mode_switch, image_timestamp);
	if (m_navigation_mode == NavigationMode::IntersectionForward) {
		return m_rejoin_distance - distance;
	} else if (m_navigation_mode == NavigationMode::IntersectionLeft || m_navigation_mode == NavigationMode::IntersectionRight) {
		// Check the angle change relative to the target 90° angle
		float angle = transform_to_sXYZ_euler(transform)[2];
		if (std::abs(angle) > 0.8f * (M_PI / 2.0f))
			return -1;  // FIXME ? Actually the rest just needs to know if the distance is passed or not
		else
			return 1;
	}
}
