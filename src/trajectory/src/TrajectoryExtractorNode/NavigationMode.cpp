#include "trajectory/TrajectoryExtractorNode.h"


//                 ╔═══════════════════════════════════╗                 #
// ════════════════╣ NAVIGATION MODE SWITCH PROCEDURES ╠════════════════ #
//                 ╚═══════════════════════════════════╝                 #

/** Base procedure for all navigation mode switches */
void TrajectoryExtractorNode::switch_navigation_mode(NavigationMode new_mode, ros::Time timestamp) {
	ROS_INFO("Navigation mode switch : %s ⟶ %s", std::to_string(m_navigation_mode).c_str(), std::to_string(new_mode).c_str());
	m_navigation_mode = new_mode;
	m_last_mode_switch = timestamp;
}

/** Switch to cruise navigation mode and set the default values for cruise */
void TrajectoryExtractorNode::switch_cruise(ros::Time image_timestamp) {
	switch_navigation_mode(NavigationMode::Cruise, image_timestamp);
	m_next_direction = Direction::Forward;
	m_last_lane_rejoin = image_timestamp;
	m_next_direction = false;
}

/** Switch to an intersection navigation mode, and compute the necessary trajectories */
void TrajectoryExtractorNode::switch_intersection(NavigationMode mode, ros::Time image_timestamp, float intersection_distance) {
	assert(mode.is_intersection());

	switch_navigation_mode(mode, image_timestamp);

	switch (mode) {
		case NavigationMode::IntersectionForward: build_intersection_forward_trajectory(image_timestamp, intersection_distance); break;
		case NavigationMode::IntersectionLeft:    build_intersection_left_trajectory(image_timestamp, intersection_distance);    break;
		case NavigationMode::IntersectionRight:   build_intersection_right_trajectory(image_timestamp, intersection_distance);   break;
		default: switch_panic(NavigationMode::PanicInvalid, image_timestamp, "Invalid navigation mode for switch_intersection : " + std::to_string(mode)); break;
	}

	// Discard all intersection hints, those won’t be useful on the other side
	m_intersection_hints.clear();

	// Clear the trajectory buffer, the old trajectories won’t be relevant anymore after the intersection
	m_trajectory_history.clear();

	// Publish the intersection navigation trajectory
	publish_trajectory(m_current_trajectory);
}

/** Switch into some panic mode, in case something goes unrecoverably wrong
  * - navigation_mode : NavigationMode : Panic navigation mode to apply
  * - image_timestamp : ros::Time      : Mode switch timestamp
  * - exception       : std::string    : Message to display */
void TrajectoryExtractorNode::switch_panic(NavigationMode mode, ros::Time image_timestamp, std::string exception) {
	switch_navigation_mode(mode, image_timestamp);
	ROS_ERROR("!!!! %s", mode.what().c_str());
	ROS_ERROR("     ------> %s", exception.c_str());

	// TODO : Stop the car
}

bool TrajectoryExtractorNode::is_panic() const {
	return m_navigation_mode.is_panic();
}
