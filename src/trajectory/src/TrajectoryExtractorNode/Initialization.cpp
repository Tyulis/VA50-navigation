#include "transformtrack/DropVelocity.h"
#include "transformtrack/TransformBatch.h"
#include "trajectory/Trajectory.h"
#include "trajectory/TrajectoryExtractorNode.h"


TrajectoryExtractorNode::TrajectoryExtractorNode() : m_tf_listener(m_tf_buffer) {
	init_fuzzy_system();
}

/* Initialize the node and spin it */
void TrajectoryExtractorNode::run() {
	ROS_INFO("Initializing...");

	// Intersection management
	m_next_direction = Direction::Forward;
	m_navigation_mode = NavigationMode::Cruise;

	// Trajectory buffers and data
	m_trajectory_seq = 0;

	// Meta stats
	m_total_duration_ms = 0.0;
	m_durations_count = 0;

	// Initialize service connections
	ROS_INFO("Waiting for the transform service...");
	ros::service::waitForService(config::node::transform_service_name);
	ros::service::waitForService(config::node::drop_service_name);
	m_transform_service = m_node.serviceClient<transformtrack::TransformBatch>(config::node::transform_service_name, true);
	m_drop_service = m_node.serviceClient<transformtrack::DropVelocity>(config::node::drop_service_name, true);

	// Initialize the topic subscribers (last to avoid early messages while waiting for other things to complete)
	m_image_subscriber = m_node.subscribe(config::node::image_topic, 1, &TrajectoryExtractorNode::callback_image, this);
	m_camerainfo_subscriber = m_node.subscribe(config::node::camerainfo_topic, 10, &TrajectoryExtractorNode::callback_camerainfo, this);
	m_direction_subscriber = m_node.subscribe(config::node::direction_topic, 10, &TrajectoryExtractorNode::callback_direction, this);
	m_trafficsigns_subscriber = m_node.subscribe(config::node::traffic_sign_topic, 10, &TrajectoryExtractorNode::callback_trafficsigns, this);
	m_trajectory_publisher = m_node.advertise<trajectory::Trajectory>(config::node::trajectory_topic, 10);
	
	ROS_INFO("Ready");
	ros::spin();
}


// Armadillo makes this unnecessarily cumbersome
void TrajectoryExtractorNode::init_fuzzy_system() {
	arma::fmat centers(5, 3);
	centers.row(0) = arma::frowvec(&config::fuzzy_lines::centers::forward_distance[0], 3);
	centers.row(1) = arma::frowvec(&config::fuzzy_lines::centers::line_distance[0], 3);
	centers.row(2) = arma::frowvec(&config::fuzzy_lines::centers::line_lengths[0], 3);
	centers.row(3) = arma::frowvec(&config::fuzzy_lines::centers::parallel_distances[0], 3);
	centers.row(4) = arma::frowvec(&config::fuzzy_lines::centers::parallel_angles[0], 3);

	arma::imat malus(5, 3);
	malus.row(0) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::forward_distance[0], 3));
	malus.row(1) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::line_distance[0], 3));
	malus.row(2) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::line_lengths[0], 3));
	malus.row(3) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::parallel_distances[0], 3));
	malus.row(4) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::parallel_angles[0], 3));
	
	arma::fvec output_centers(&config::fuzzy_lines::centers::output[0], 5);

	m_lane_fuzzysystem = FuzzyLaneSystem(centers, malus, output_centers, config::fuzzy_lines::base_score);
}