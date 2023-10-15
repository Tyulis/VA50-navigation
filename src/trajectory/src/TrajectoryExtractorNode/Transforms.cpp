#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/TransformStamped.h"

#include "transformtrack/DropVelocity.h"
#include "transformtrack/TransformBatch.h"
#include "trajectory/TrajectoryExtractorNode.h"


//               ╔═══════════════════════════════════════╗               #
// ══════════════╣ TRANSFORM MANAGEMENT AND MEASUREMENTS ╠══════════════ #
//               ╚═══════════════════════════════════════╝               #



/** Get the latest transform matrix from `source_frame` to `target_frame`
  * - source_frame : std::string      : Name of the source frame
  * - target_frame : std::string      : Name of the target frame
  * <------------- : arma::fmat[4, 4] : 3D homogeneous transform matrix to convert from `source_frame` to `target_frame`,
		                            	or an empty matrix if no TF for those frames was published */
arma::fmat TrajectoryExtractorNode::get_transform(std::string const& source_frame, std::string const& target_frame) {
	geometry_msgs::TransformStamped message;
	try {
		message = m_tf_buffer.lookupTransform(target_frame, source_frame, ros::Time(0));
	} catch (tf2::TransformException &exc) {
		ROS_ERROR("Failed to retrieve the transform from %s to %s :\n\t%s", source_frame.c_str(), target_frame.c_str(), exc.what());
		return arma::fmat();
	}

	arma::fmat transform(4, 4, arma::fill::eye);
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

	// Translation vector
	transform(0, 3) = static_cast<float>(message.transform.translation.x);
	transform(1, 3) = static_cast<float>(message.transform.translation.y);
	transform(2, 3) = static_cast<float>(message.transform.translation.z);
	return transform;
}

/** Get a batch of transforms of the vehicle frame from the given `start_times` to `end_time`
  * - start_times : std::vector<ros::Time> : Timestamps to get the transforms from
  * - end_time    : ros::Time              : Target timestamp, to get the transforms to
  * <-------------- arma::fcube[4, 4, N]   : 3D homogeneous transform matrices to transform points in the vehicle frame
											 at `start_times[i]` to the vehicle frame at `end_time`
  * */		
std::tuple<arma::fcube, arma::frowvec> TrajectoryExtractorNode::get_map_transforms(std::vector<ros::Time> start_times, ros::Time end_time) {
	// Build the request to the transform service, see transformtrack/srv/TransformBatch.srv and transformtrack/msg/TimeBatch.msg for info
	transformtrack::TransformBatch::Request request;
	transformtrack::TransformBatch::Response response;
	request.timestamps.start_times = start_times;
	request.timestamps.end_time = end_time;
	
	// Now send the request and get the response
	// We use persistent connections to improve efficiency, and ROS advises to implement some reconnection logic
	// in case the network gets in the way, so in case of disconnection, retry 10 times to reconnect then fail
	int tries = 0;
	while (true) {
		// Apparently, when a call to the service is pending, the node is free to serve other callbacks,
		// including callback_trafficsign that also calls this service
		// So with the traffic signs subscriber active, it’s only a matter of time until both get to their transform service call concurrently
		// For some reason, ROS allows it, and for some reason it deadlocks ROS as a whole
		// So let’s throw in a lock to prevent ROS from killing itself
		m_transform_service_mutex.lock();
		bool success = m_transform_service.call(request, response);
		m_transform_service_mutex.unlock();

		if (success)
			break;
		
		if (tries >= 10) {
			ROS_ERROR("Connection to service %s failed %d times, skipping", config::node::transform_service_name, tries);
			return arma::fcube();
		}

		// Try to reconnect
		m_transform_service.shutdown();
		m_transform_service = m_node.serviceClient<transformtrack::TransformBatch>(config::node::transform_service_name, true);
		tries += 1;
	}

	// The call was successful, get the transforms in the right format and return
	// FIXME : This is for compatibility with the former Python interface
	arma::dcube transforms64(response.transforms.data.data(), response.transforms.layout.dim[2].size, response.transforms.layout.dim[1].size, response.transforms.layout.dim[0].size, false);
	arma::fcube transforms(transforms64);
	arma::frowvec distances(response.distances);
	return transforms, distances;
}

/** Convenience wrapper for get_map_transforms with a single start timestamp */
std::tuple<arma::fmat, float> TrajectoryExtractorNode::get_map_transforms(ros::Time start_time, ros::Time end_time) {
	std::vector<ros::Time> start_times = {start_time};
	auto [transforms, distances] =  get_map_transforms(start_times, end_time);
	return transforms.slice(0), distances(0);
}

/** Call the DropVelocity service, such that the TransformBatch service discards its old velocity data
  * and doesn’t unnecessarily clutter its memory and performance
  * - end_time : rospy.Time : Discard all velocity data prior to this timestamp */
void TrajectoryExtractorNode::drop_velocity(ros::Time end_time) {
	// Build the request to the transform service, see transformtrack/srv/TransformBatch.srv and transformtrack/msg/TimeBatch.msg for info
	transformtrack::DropVelocity::Request request;
	transformtrack::DropVelocity::Response response;
	request.end_time = end_time;
	
	// Now send the request and get the response
	// We use persistent connections to improve efficiency, and ROS advises to implement some reconnection logic
	// in case the network gets in the way, so in case of disconnection, retry 10 times to reconnect then fail
	int tries = 0;
	while (true) {
		// Apparently, when a call to the service is pending, the node is free to service other callbacks,
		// including callback_trafficsign that also call this service
		// So with the traffic signs subscriber active, it’s only a matter of time until both get to their transform service call concurrently
		// For some reason, ROS allows it, and for some reason it deadlocks ROS as a whole
		// So let’s throw in a lock to prevent ROS from killing itself
		m_transform_service_mutex.lock();
		bool success = m_drop_service.call(request, response);
		m_transform_service_mutex.unlock();

		if (success)
			break;
		
		if (tries >= 10) {
			ROS_ERROR("Connection to service %s failed %d times, skipping", config::node::drop_service_name, tries);
			return;
		}

		// Try to reconnect
		m_drop_service.shutdown();
		m_drop_service = m_node.serviceClient<transformtrack::DropVelocity>(config::node::drop_service_name, true);
		tries += 1;
	}
}
