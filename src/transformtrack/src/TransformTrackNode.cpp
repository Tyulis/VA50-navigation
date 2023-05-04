//   Copyright 2023 Grégori MIGNEROT, Élian BELMONTE, Benjamin STACH
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include <algorithm>

#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/MultiArrayDimension.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/TransformStamped.h"

#include "transformtrack/TransformTrackNode.h"



TransformTrackNode::TransformTrackNode() : m_tf_listener(m_tf_buffer) {
	
}

void TransformTrackNode::run() {
	ROS_INFO("Initializing...");

	m_velocity_subscriber = m_node.subscribe(config::node::velocity_topic, 10, &TransformTrackNode::callback_velocity, this);
	m_transform_batch_service = m_node.advertiseService(config::node::transform_service_name, &TransformTrackNode::handle_transform_batch, this);
	m_drop_velocity_service = m_node.advertiseService(config::node::drop_service_name, &TransformTrackNode::handle_drop_velocity, this);

	m_transform_manager = TransformManager(config::transform::sim_interval);
	
	ROS_INFO("Ready");
	ros::spin();
}

void TransformTrackNode::callback_velocity(geometry_msgs::TwistStamped::ConstPtr const& message) {
	arma::dvec linear_velocity = {message->twist.linear.x, message->twist.linear.y, message->twist.linear.z};
	arma::dvec angular_velocity = {message->twist.angular.x, message->twist.angular.y, message->twist.angular.z};
	m_transform_manager.add_velocity(message->header.stamp.toSec(), linear_velocity, angular_velocity);
}

bool TransformTrackNode::handle_transform_batch(transformtrack::TransformBatch::Request& request, transformtrack::TransformBatch::Response& response) {
	// Retrieve the timestamps
	arma::dvec start_timestamps(request.timestamps.start_times.size());
	std::transform(request.timestamps.start_times.begin(), request.timestamps.start_times.end(), start_timestamps.begin(), [](ros::Time timestamp) {return timestamp.toSec();});
	double end_timestamp = request.timestamps.end_time.toSec();
	
	// Compute the transforms
	arma::fcube transforms = m_transform_manager.get_map_transforms(start_timestamps, end_timestamp);

	// Those transforms are from start_time to end_time, expressed in the map frame as the velocity refers to the map frame
	// So we need to rotate those back to the end_time frame
	arma::fmat map_transform = get_rotation(config::node::world_frame, config::node::road_frame);
	transforms.each_slice([&](arma::fmat& transform) {
		transform.submat(0, 3, 2, 3) = map_transform * transform.submat(0, 3, 2, 3);
		transform.submat(0, 0, 2, 2) = map_transform * transform.submat(0, 0, 2, 2) * map_transform.t();
	}, true);

	// And build the response
	response.transforms.data.insert(response.transforms.data.end(), transforms.begin(), transforms.end());
	
	response.transforms.layout.dim.emplace_back();
	response.transforms.layout.dim[0].size = transforms.n_slices;
	response.transforms.layout.dim[0].stride = transforms.n_slices * transforms.n_cols * transforms.n_rows;
	response.transforms.layout.dim[0].label = "transforms";

	response.transforms.layout.dim.emplace_back();
	response.transforms.layout.dim[1].size = transforms.n_cols;
	response.transforms.layout.dim[1].stride = transforms.n_cols * transforms.n_rows;
	response.transforms.layout.dim[1].label = "columns";

	response.transforms.layout.dim.emplace_back();
	response.transforms.layout.dim[2].size = transforms.n_rows;
	response.transforms.layout.dim[2].stride = transforms.n_rows;
	response.transforms.layout.dim[2].label = "rows";

	return true;
}

bool TransformTrackNode::handle_drop_velocity(transformtrack::DropVelocity::Request& request, transformtrack::DropVelocity::Response& response) {
	double end_timestamp = request.end_time.toSec();
	m_transform_manager.drop_velocity(end_timestamp);
	response.done = true;
	return true;
}


/** Get the latest rotation matrix from `source_frame` to `target_frame`
  * - source_frame : std::string      : Name of the source frame
  * - target_frame : std::string      : Name of the target frame
  * <------------- : arma::fmat[3, 3] : 3D rotation matrix to convert from `source_frame` to `target_frame`,
	                                    or None if no TF for those frames was published */
arma::fmat TransformTrackNode::get_rotation(std::string const& source_frame, std::string const& target_frame) {
	arma::fmat transform(3, 3, arma::fill::eye);
	geometry_msgs::TransformStamped message;
	try {
		message = m_tf_buffer.lookupTransform(target_frame, source_frame, ros::Time(0));
	} catch (tf2::TransformException &exc) {
		ROS_ERROR("Failed to retrieve the transform from %s to %s :\n\t%s", source_frame.c_str(), target_frame.c_str(), exc.what());
		return transform;  // ???
	}

	geometry_msgs::Quaternion rotation = message.transform.rotation;
	
	// Formula for the rotation matrix from a quaternion from here :
	// https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
	float s = static_cast<float>(1.0 / (rotation.w*rotation.w + rotation.x*rotation.x + rotation.y*rotation.y + rotation.z*rotation.z));
	
	// Rotation matrix from the resulting quaternion
	transform(0, 0) = static_cast<float>(1 - 2*s * (rotation.y * rotation.y + rotation.z * rotation.z));
	transform(1, 0) = static_cast<float>(    2*s * (rotation.x * rotation.y + rotation.z * rotation.w));
	transform(2, 0) = static_cast<float>(    2*s * (rotation.x * rotation.z - rotation.y * rotation.w));
	transform(0, 1) = static_cast<float>(    2*s * (rotation.x * rotation.y - rotation.z * rotation.w));
	transform(1, 1) = static_cast<float>(1 - 2*s * (rotation.x * rotation.x + rotation.z * rotation.z));
	transform(2, 1) = static_cast<float>(    2*s * (rotation.y * rotation.z + rotation.x * rotation.w));
	transform(0, 2) = static_cast<float>(    2*s * (rotation.x * rotation.z + rotation.y * rotation.w));
	transform(1, 2) = static_cast<float>(    2*s * (rotation.y * rotation.z - rotation.x * rotation.w));
	transform(2, 2) = static_cast<float>(1 - 2*s * (rotation.x * rotation.x + rotation.y * rotation.y));
	return transform;
}