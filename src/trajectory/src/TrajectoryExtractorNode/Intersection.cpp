#include <mlpack/methods/dbscan.hpp>

#include "trajectory/Utility.h"
#include "trajectory/TrajectoryExtractorNode.h"


//                 ╔═══════════════════════════════════╗                 #
// ════════════════╣      INTERSECTION MANAGEMENT      ╠════════════════ #
//                 ╚═══════════════════════════════════╝                 #

/** Add a new intersection hint, that may be added to the list of hints,
 *  or merged with an existing hint if they are deemed different measurements of the same object
 *  - hint : IntersectionHint : New hint to consider */
void TrajectoryExtractorNode::add_intersection_hint(IntersectionHint const& hint) {
	// Skip hints found too close to the last intersection
	if (m_last_lane_rejoin.isZero()) {
		arma::fmat transform = get_map_transforms(m_last_lane_rejoin, hint.timestamps.back());
		float distance = arma::norm(transform(arma::span(0, 2), 3));  // Norm of the translation vector
		
		// Too close to the previous intersection, skip
		if (distance < config::intersection::hint_detection_buffer)
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
	arma::fcube transforms = get_map_transforms(existing_hint.timestamps, hint.timestamps.back());
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
			// The intersection does not allow the chosen direction : go forth, keep for next time
			if (intersection_directions & Direction::Forward)
				switch_intersection(NavigationMode::IntersectionForward, image_timestamp, intersection_distance);
			
			// Only left or right, no direction chosen : panic, wait for input
			else
				switch_panic(NavigationMode::PanicNoDirection, image_timestamp, "No direction chosen");
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
 *  <------------------ float     : Distance to the next intersection
 *  <------------------ Direction : Directions available at the next intersection */
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
	arma::fcube transforms = get_map_transforms(hint_timestamps, image_timestamp);
	arma::fmat transformed = transform_positions(transforms, hint_positions);

	// Project everything onto the current directional vector (0, 1, 0) and disregard points that are behind the vehicle
	arma::dmat current_distances = arma::conv_to<arma::dmat>::from(transformed.row(1));
	arma::uvec forward_filter = arma::find(current_distances >= 0);
	current_distances = current_distances(forward_filter);
	hint_indices = hint_indices(forward_filter);

	// Less than two hints : just take the closest one
	if (current_distances.n_elem <= 2) {
		arma::uword selected_index = current_distances.index_min();
		IntersectionHint hint = m_intersection_hints[hint_indices(selected_index)];
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
		return {current_distances(selected_index), hint.direction_hint()};
	}

	// Take the cluster associated with the closest centroid
	arma::uword selected_cluster = centroids.index_min();
	float intersection_distance = centroids[selected_cluster];
	arma::uvec selected_positions = arma::find(assignments == selected_cluster);
	Direction intersection_directions = Direction::All;
	arma::unique(hint_indices(selected_positions)).eval().for_each([&] (int const& hint_index) {
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
	arma::fmat transform = get_map_transforms(m_last_mode_switch, image_timestamp);
	float distance = arma::norm(transform(arma::span(0, 2), 3));
	return m_rejoin_distance - distance;
}
