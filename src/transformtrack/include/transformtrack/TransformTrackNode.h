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

#ifndef _TRANSFORMTRACK_TRANSFORMTRACKNODE_H
#define _TRANSFORMTRACK_TRANSFORMTRACKNODE_H

#include "ros/ros.h"
#include "tf2_ros/transform_listener.h"
#include "geometry_msgs/TwistStamped.h"

#include "config.h"
#include "transformtrack/TimeBatch.h"
#include "transformtrack/DropVelocity.h"
#include "transformtrack/TransformBatch.h"
#include "transformtrack/TransformManager.h"


class TransformTrackNode {
	public:
		TransformTrackNode();

		void run();

		void callback_velocity(geometry_msgs::TwistStamped::ConstPtr const& message);
		bool handle_transform_batch(transformtrack::TransformBatch::Request& request, transformtrack::TransformBatch::Response& response);
		bool handle_drop_velocity(transformtrack::DropVelocity::Request& request, transformtrack::DropVelocity::Response& response);

	private:
		arma::fmat get_rotation(std::string const& source_frame, std::string const& target_frame);

		ros::NodeHandle m_node;
		ros::Subscriber m_velocity_subscriber;
		ros::ServiceServer m_transform_batch_service;
		ros::ServiceServer m_drop_velocity_service;

		tf2_ros::Buffer m_tf_buffer;
		tf2_ros::TransformListener m_tf_listener;

		TransformManager m_transform_manager;
};

#endif /* _TRANSFORMTRACK_TRANSFORMTRACKNODE_H */