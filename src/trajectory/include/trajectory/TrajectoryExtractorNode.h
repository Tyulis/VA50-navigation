#ifndef _TRAJECTORY_TRAJECTORYEXTRACTORNODE_H
#define _TRAJECTORY_TRAJECTORYEXTRACTORNODE_H

#include <tuple>
#include <thread>
#include <vector>
#include <string>
#include <armadillo>
#include <opencv2/opencv.hpp>

#include "ros/ros.h"
#include "tf2_ros/transform_listener.h"

#include "std_msgs/UInt8.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "trafficsigns/TrafficSign.h"
#include "trafficsigns/TrafficSignStatus.h"

#include "config.h"
#include "trajectory/Direction.h"
#include "trajectory/DiscreteCurve.h"
#include "trajectory/NavigationMode.h"
#include "trajectory/FuzzyLaneSystem.h"
#include "trajectory/IntersectionHint.h"
#include "trajectory/TrajectoryVisualizer.h"


/* Main class for the ROS node that manages the trajectory */
class TrajectoryExtractorNode {
	public:
		TrajectoryExtractorNode();

		void run();
		void init_fuzzy_system();

		// Callbacks, ROS direct interactions and global activity procedures
		void callback_image(sensor_msgs::Image::ConstPtr const& message);
		void callback_camerainfo(sensor_msgs::CameraInfo::ConstPtr const& message);
		void callback_direction(std_msgs::UInt8::ConstPtr const& message);
		void callback_trafficsigns(trafficsigns::TrafficSignStatus::ConstPtr const& message);
		void publish_trajectory(DiscreteCurve const& trajectory);
		void extract_trajectory(cv::Mat& image, ros::Time timestamp, std::string camera_frame);
	
	private:
		// Transform management and measurement
		arma::fmat get_transform(std::string const& source_frame, std::string const& target_frame);
		std::tuple<arma::fcube, arma::frowvec> get_map_transforms(std::vector<ros::Time> start_times, ros::Time end_time);
		std::tuple<arma::fmat, float> get_map_transforms(ros::Time start_time, ros::Time end_time);
		void drop_velocity(ros::Time end_time);

		// Intersection management
		void add_intersection_hint(IntersectionHint const& hint);
		bool match_intersection_hint(IntersectionHint const& existing_hint, IntersectionHint const& hint);
		void update_intersection(ros::Time image_timestamp);
		std::tuple<float, Direction> next_intersection(ros::Time image_timestamp);
		float distance_until_rejoin(ros::Time image_timestamp);

		// Navigation mode switch procedures
		void switch_navigation_mode(NavigationMode mode, ros::Time image_timestamp);
		void switch_cruise(ros::Time image_timestamp);
		void switch_intersection(NavigationMode mode, ros::Time image_timestamp, float intersection_distance);
		void switch_panic(NavigationMode mode, ros::Time image_timestamp, std::string exception);
		bool is_panic() const;

		// Image processing
		std::tuple<cv::Mat, cv::Mat, float> preprocess_image(cv::Mat& image, arma::fmat const& target_to_camera);

		// Markings detection
		std::tuple<DiscreteCurve, DiscreteCurve> detect_lane(std::vector<DiscreteCurve>& lines, float scale_factor, ros::Time image_timestamp, std::vector<Marking> const& markings);
		std::tuple<int, int, float, float> detect_full_lane(arma::fvec const& forward_distance, arma::fvec const& left_line_distance, arma::fvec const& right_line_distance,
		                                                    arma::fvec const& line_lengths, arma::fmat const& parallel_distances, arma::fmat const& parallel_angles, bool fallback);
		std::tuple<int, int, float, float> detect_any_line(arma::fvec const& forward_distance, arma::fvec const& left_line_distance, arma::fvec const& right_line_distance, arma::fvec const& line_lengths);
		void init_lane_scores(DiscreteCurve& line, float base_score);

		// Trajectory building
		void add_local_trajectory(DiscreteCurve const& trajectory);
		void compile_trajectory(ros::Time timestamp, DiscreteCurve& left_line, DiscreteCurve right_line, cv::Mat& viz);
		std::vector<DiscreteCurve> pull_trajectories(ros::Time timestamp);
		float turn_radius(ros::Time timestamp, float intersection_distance);
		void build_intersection_forward_trajectory(ros::Time image_timestamp, float intersection_distance);
		void build_intersection_left_trajectory(ros::Time image_timestamp, float intersection_distance);
		void build_intersection_right_trajectory(ros::Time image_timestamp, float intersection_distance);
		void update_trajectory(ros::Time timestamp, cv::Mat& viz);

		// Visualization
		void viz_intersection_mode(cv::Mat& viz, float scale_factor, ros::Time image_timestamp, float remaining_distance);


		// ROS interaction
		ros::NodeHandle m_node;
		ros::Subscriber m_image_subscriber;
		ros::Subscriber m_camerainfo_subscriber;
		ros::Subscriber m_direction_subscriber;
		ros::Subscriber m_trafficsigns_subscriber;
		ros::Publisher m_trajectory_publisher;
		ros::ServiceClient m_transform_service;
		ros::ServiceClient m_drop_service;
		std::mutex m_transform_service_mutex;

		// TF stuff
		tf2_ros::Buffer m_tf_buffer;
		tf2_ros::TransformListener m_tf_listener;

		// Camera info
		arma::fmat m_camera_to_image;
		float m_distortion_xi;

		// Intersection management
		NavigationMode m_navigation_mode;
		Direction m_next_direction;
		bool m_next_double_lane;
		ros::Time m_last_lane_rejoin;
		ros::Time m_last_mode_switch;
		float m_rejoin_distance;
		std::vector<IntersectionHint> m_intersection_hints;

		// Lane detection
		FuzzyLaneSystem m_lane_fuzzysystem;

		// Trajectory buffers
		std::vector<DiscreteCurve> m_trajectory_history;
		std::vector<DiscreteCurve> m_local_trajectory_history;
		DiscreteCurve m_current_trajectory;
		int m_trajectory_seq;

		// Meta stats
		double m_total_duration_ms;
		int m_durations_count;

		// Visualization
		TrajectoryVisualizer m_visualization;
};

#endif