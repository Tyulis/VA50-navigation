#include <chrono>
#include <boost/algorithm/string.hpp>
#include <frozen/unordered_set.h>
#include <frozen/string.h>

#include "transformtrack/DropVelocity.h"
#include "transformtrack/TransformBatch.h"
#include "trajectory/Marking.h"
#include "trajectory/Utility.h"
#include "trajectory/Trajectory.h"
#include "trajectory/LineDetection.h"
#include "trajectory/TrajectoryExtractorNode.h"

//                        ╔══════════════════════╗                       #
// ═══════════════════════╣ SUBSCRIBER CALLBACKS ╠══════════════════════ #
//                        ╚══════════════════════╝                       #


static constexpr frozen::unordered_set<frozen::string, 9> INTERSECTION_SIGNS = {
	"yield", "stop",
	"right-only", "left-only", "ahead-only",
	"straight-right-only", "straight-left-only",
	"keep-right", "keep-left",
};

static constexpr frozen::unordered_set<frozen::string, 7> TURN_SIGNS = {
	"right-only", "left-only", "ahead-only",
	"straight-right-only", "straight-left-only",
	"keep-right", "keep-left",
};


/** Callback called when an image is published from the camera
  * - message : sensor_msgs::Image : Message from the camera */
void TrajectoryExtractorNode::callback_image(sensor_msgs::Image::ConstPtr const& message) {
	if (is_panic())
		return;
	
	if (m_camera_to_image.is_empty()) {
		ROS_WARN("Image dropped : no valid CameraInfo");
		return;
	}

	ROS_INFO("---------- Received an image");
	// The data is not copied. Is it safe to keep the message data that way ?
	unsigned char* data = new unsigned char[message->height * message->step];
	std::memcpy(data, message->data.data(), sizeof(unsigned char) * message->height * message->step);
	cv::Mat image(message->height, message->width, CV_8UC3, data, message->step);

	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
	extract_trajectory(image, message->header.stamp, message->header.frame_id);
	std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

	double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
	m_total_duration_ms += milliseconds;
	m_durations_count += 1;

	delete[] data;

	ROS_INFO("Image handled in %f ms, mean is %f ms per image", milliseconds, m_total_duration_ms / m_durations_count);
}

/** Callback called when a new camera info message is published
  * - message : sensor_msgs::CameraInfo : Message with metadata about the camera */
void TrajectoryExtractorNode::callback_camerainfo(sensor_msgs::CameraInfo::ConstPtr const& message) {
	if (boost::algorithm::to_lower_copy(message->distortion_model) != "mei") {
		ROS_ERROR("Unsupported distortion model : %s", message->distortion_model.c_str());
		return;
	}

	m_camera_to_image = {{float(message->P[0]), float(message->P[1]), float(message->P[2])},
	                     {float(message->P[4]), float(message->P[5]), float(message->P[6])},
						 {float(message->P[8]), float(message->P[9]), float(message->P[10])}};
	m_distortion_xi = message->D[0];
}

/** Callback called when a direction is sent from the navigation nodes
  * - message : std_msgs::Uint8 : Message with the direction (same values as Direction::Forward, ::Left and ::Right) */
void TrajectoryExtractorNode::callback_direction(std_msgs::UInt8::ConstPtr const& message) {
	Direction direction(message->data);
	switch (direction) {
		case Direction::Forward:
			ROS_INFO("Updated next direction to FORWARD");
			m_next_direction = Direction::Forward;
			break;
		case Direction::Left:
			ROS_INFO("Updated next direction to LEFT");
			m_next_direction = Direction::Left;
			break;
		case Direction::Right:
			ROS_INFO("Updated next direction to RIGHT");
			m_next_direction = Direction::Right;
			break;
		case Direction::DoubleLane:
			ROS_INFO("Next intersection has a double lane");
			m_next_double_lane = true;
			break;
		case Direction::ForceIntersection:
			ROS_INFO("Force intersection mode");
			switch (m_next_direction) {
				case Direction::Forward: switch_intersection(NavigationMode::IntersectionForward, ros::Time::now(), 0);  break;
				case Direction::Left   : switch_intersection(NavigationMode::IntersectionLeft,    ros::Time::now(), 0);  break;
				case Direction::Right  : switch_intersection(NavigationMode::IntersectionRight,   ros::Time::now(), 0);  break;
			}
			break;
		default:
			ROS_ERROR("Invalid direction ID received : %d", message->data);
			return;
	}
}

/** Callback called when traffic signs are detected and received
  * - message : trafficsigns::TrafficSignStatus : Message with the detected traffic signs data */
void TrajectoryExtractorNode::callback_trafficsigns(trafficsigns::TrafficSignStatus::ConstPtr const& message) {
	for (auto it = message->traffic_signs.begin(); it != message->traffic_signs.end(); it++) {
		// FIXME : The python version uses TURN_SIGNS ...?
		if (INTERSECTION_SIGNS.contains(frozen::string(it->type)) && it->confidence > config::intersection::confidence_threshold::trafficsign) {
			arma::fvec position = {it->x, it->y, it->z};
			IntersectionHint hint(IntersectionHint::Category::TrafficSign, it->type, position, message->header.stamp, it->confidence);
			add_intersection_hint(hint);
		}
	}
}


/** Publish a trajectory on the output topic
  * - trajectory : Trajectory : Trajectory to publish */
void TrajectoryExtractorNode::publish_trajectory(DiscreteCurve const& trajectory) {
	trajectory::Trajectory message;
	message.header.seq = m_trajectory_seq++;
	message.header.stamp = trajectory.timestamp;
	message.header.frame_id = config::node::road_frame;

	message.trajectory.data.insert(message.trajectory.data.end(), trajectory.curve.begin(), trajectory.curve.end());
	
	message.trajectory.layout.dim.emplace_back();
	message.trajectory.layout.dim[0].size = trajectory.curve.n_cols;
	message.trajectory.layout.dim[0].stride = trajectory.curve.n_cols * trajectory.curve.n_rows;
	message.trajectory.layout.dim[0].label = "columns";

	message.trajectory.layout.dim.emplace_back();
	message.trajectory.layout.dim[1].size = trajectory.curve.n_rows;
	message.trajectory.layout.dim[1].stride = trajectory.curve.n_rows;
	message.trajectory.layout.dim[1].label = "rows";

	ROS_INFO("Publishing a new trajectory, lag is %.3f seconds", (ros::Time::now() - trajectory.timestamp).toSec());
	m_trajectory_publisher.publish(message);
}


/** Global procedure called each time an image is received
  * Take the image received from the camera, estimate the trajectory from it and publish it if necessary
  * - image       : cv::Mat[y, x, CV_8UC3] : RGB image received from the camera
  * - timestamp   : ros::Time              : Timestamp at which the image has been taken. All operations will attune to this timestamp
  * - image_frame : std::string            : Name of the TF frame of the camera that has taken the picture */
void TrajectoryExtractorNode::extract_trajectory(cv::Mat& image, ros::Time timestamp, std::string camera_frame) {
	// Get the transform from the local vehicle frame (base_link) to the camera
	arma::fmat target_to_camera = get_transform(config::node::road_frame, camera_frame);
	if (target_to_camera.is_empty()) {
		ROS_ERROR("No camera -> road transform found");
		return;
	}

	// Pull the trajectory history to the current local frame
	m_local_trajectory_history = pull_trajectories(timestamp);

	// Preprocess the image
	auto [birdeye, be_binary, scale_factor] = preprocess_image(image, target_to_camera);

	// The following is made overly complicated because of the visualization that must still be updated even when nothing else is going on
	// First set up the HSV trajectory viz frame with the grayscale bird-eye view as background
	cv::Mat trajectory_viz;
	if (config::node::visualize) {
		trajectory_viz = cv::Mat(birdeye.rows, birdeye.cols, CV_8UC3);
		cv::cvtColor(birdeye, trajectory_viz, cv::COLOR_GRAY2BGR);
		cv::cvtColor(trajectory_viz, trajectory_viz, cv::COLOR_BGR2HSV);
		m_visualization.update_background(be_binary);
	}

	// Intersection mode : Check whether the vehicle has reached the rejoin distance, and if so, try to catch the new lane
	// Otherwise just wait further
	float remaining_distance = -1;
	bool must_build_trajectory = !m_navigation_mode.is_panic();
	if (m_navigation_mode.is_intersection()) {
		remaining_distance = distance_until_rejoin(timestamp);
		must_build_trajectory &= remaining_distance <= 0;
		if (remaining_distance > 0) {
			ROS_INFO("Waiting for rejoin, %f meters remaining", remaining_distance);
			viz_intersection_mode(trajectory_viz, scale_factor, timestamp, remaining_distance);
		}
	}

	// In cruise mode or intersection mode when the vehicle needs to catch the new lane, go forth with the detection
	if (must_build_trajectory) {
		std::vector<DiscreteCurve> branches = extract_branches(be_binary);
		std::vector<Marking> markings = detect_markings(branches, scale_factor);
		for (auto it = markings.begin(); it != markings.end(); it++) {
			if (it->type == Marking::Type::Crosswalk) {
				arma::fvec marking_centroid = arma::mean(arma::mean(it->data, 1), 2);
				IntersectionHint hint(IntersectionHint::Category::Marking, "crosswalk", arma::fvec{marking_centroid(0), marking_centroid(1), 0}, timestamp, it->confidence);
				add_intersection_hint(hint);
			}
		}

		std::vector<DiscreteCurve> filtered_lines = filter_lines(branches, scale_factor);

		// Flip the lines so that they start at the bottom of the image, and convert them to metric
		std::vector<DiscreteCurve> lines;
		std::vector<DiscreteCurve> transverse_lines;
		for (auto it = filtered_lines.begin(); it != filtered_lines.end(); it++) {
			arma::fmat gradient = it->gradient();
			arma::frowvec angles = arma::atan2(gradient.row(1), gradient.row(0));
			if (arma::all(((      -config::markings::transverse_angle < angles) && (angles <        config::markings::transverse_angle)) ||
			               ((M_PI -config::markings::transverse_angle < angles) && (angles < M_PI + config::markings::transverse_angle)))) {
				transverse_lines.push_back(*it);
			// Inverted because before birdeye_to_target, the y axis is flipped
			} else if (it->curve(1, 0) < it->curve(1, it->size() - 1)) {
				it->curve = birdeye_to_target_config(arma::fliplr(std::move(it->curve)));
				lines.push_back(*it);
			} else {
				it->curve = birdeye_to_target_config(std::move(it->curve));
				lines.push_back(*it);
			}
		}

		// Detect the current lane
		auto [left_line, right_line] = detect_lane(lines, scale_factor, timestamp, markings);
		
		// In intersection mode, if a new full lane has been found, catch it and go back to cruise 
		if (m_navigation_mode.is_intersection()) {
			if (left_line.is_valid() && right_line.is_valid()) {
				ROS_INFO("Full lane detected, switching back to cruise");
				switch_cruise(timestamp);
			} else {
				ROS_INFO("No full lane to rejoin detected, waiting further");
				viz_intersection_mode(trajectory_viz, scale_factor, timestamp, remaining_distance);
			}
		}

		// In cruise mode, update the trajectory, intersection status, and publish if there is something to publish
		// Do NOT refactor this into an `else`, it must also be done when the vehicle just got out of intersection mode (`switch_cruise` above)
		if (m_navigation_mode == NavigationMode::Cruise) {
			compile_trajectory(timestamp, left_line, right_line, trajectory_viz);

			// For intersection trajectory estimation, it’s beneficial to have the most recent trajectory available,
			// so when in cruise mode, update the intersection status only after having compiled the latest trajectory
			update_intersection(timestamp);

			if (m_trajectory_history.size() > 0)
				update_trajectory(timestamp, trajectory_viz);
			
			if (m_current_trajectory.is_valid())
				publish_trajectory(m_current_trajectory);
		}
	}

	if (config::node::visualize)
		m_visualization.update_trajectory_construction(trajectory_viz);
}


/** Visualization in intersection navigation mode
  * - viz                : cv::Mat[y, x, CV_8UC3] : Bird-eye view visualization image
  * - scale_factor       : float                  : Scale factor from pixel to metric lengths
  * - image_timestamp    : ros::Time              : Timestamp to visualize at
  * - remaining_distance : float                  : Distance until reaching the rejoin distance */
void TrajectoryExtractorNode::viz_intersection_mode(cv::Mat& viz, float scale_factor, ros::Time image_timestamp, float remaining_distance) {
	auto [transform, distance] = get_map_transforms(m_current_trajectory.timestamp, image_timestamp);
	arma::fmat local_trajectory = transform_2d(transform, m_current_trajectory.curve);
	arma::imat viz_trajectory = target_to_birdeye_config(local_trajectory);

	std::vector<cv::Point> points = arma::conv_to<std::vector<cv::Point>>::from(viz_trajectory);
	cv::polylines(viz, points, false, cv::Scalar(60, 255, 255), 2);

	if (remaining_distance > 0) {
		cv::Point center(viz.cols / 2, viz.rows - 1);
		cv::circle(viz, center, int(remaining_distance / scale_factor), cv::Scalar(0, 255, 255), 1);
	}
}