#!/usr/bin/env python3

import sys
import yaml
import numpy as np
import transforms3d.quaternions as quaternions

import rospy
import tf2_ros
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from geometry_msgs.msg import TwistStamped
from circulation.srv import TransformBatch, TransformBatchResponse, DropVelocity, DropVelocityResponse
from circulation.msg import TimeBatch

import positioning

class TransformBatchServer(object):
	def __init__(self, parameters):
		self.parameters = parameters
		self.transform_service = rospy.Service(self.parameters["node"]["transform-service-name"], TransformBatch, self.handle_transform_batch)
		self.drop_service = rospy.Service(self.parameters["node"]["drop-service-name"], DropVelocity, self.handle_drop_velocity)
		
		self.transform_manager = positioning.TransformManager(self.parameters["transform"]["sim-interval"])
		
		if self.parameters["node"]["time-discrepancy"]:
			self.time_discrepancy_buffer = []
			self.time_discrepancy = None
		else:
			self.time_discrepancy = rospy.Duration(0)
		
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.velocity_subscriber = rospy.Subscriber(self.parameters["node"]["velocity-topic"], TwistStamped, self.callback_velocity, queue_size=10)
		if self.time_discrepancy is None:
			rospy.loginfo("Service initialized, waiting for time discrepancy estimation")
		else:
			rospy.loginfo("Ready")
	
	def get_rotation(self, source_frame, target_frame):
		"""Get the latest rotation matrix from `source_frame` to `target_frame`
		   - source_frame : str           : Name of the source frame
		   - target_frame : str           : Name of the target frame
		<---------------- : ndarray[3, 3] : 3D rotation matrix to convert from `source_frame` to `target_frame`,
		                                    or None if no TF for those frames was published
		"""
		try:
			transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			return None

		# Build the matrix elements from the translation vector and the rotation quaternion
		rotation_message = transform.transform.rotation
		rotation_quaternion = np.asarray((rotation_message.w, rotation_message.x, rotation_message.y, rotation_message.z))
		rotation_matrix = quaternions.quat2mat(rotation_quaternion)
		return rotation_matrix


	def callback_velocity(self, data):
		if self.time_discrepancy is None:
			self.time_discrepancy_buffer.append(rospy.get_rostime() - data.header.stamp)
			if len(self.time_discrepancy_buffer) > 5:
				self.time_discrepancy = self.time_discrepancy_buffer[-1]
				rospy.loginfo(f"Time discrepancy estimation done ({self.time_discrepancy}), ready to receive requests")
		else:
			self.transform_manager.add_velocity((data.header.stamp + self.time_discrepancy).to_sec(),
			                                    (data.twist.linear.x, data.twist.linear.y, data.twist.linear.z),
				                                (data.twist.angular.x, data.twist.angular.y, data.twist.angular.z))
	
	def handle_transform_batch(self, request):
		if request.unbias:
			if self.time_discrepancy is None:
				return None
			start_timestamps = [timestamp + self.time_discrepancy for timestamp in request.timestamps.start_times]
			end_timestamp = request.timestamps.end_time + self.time_discrepancy
		else:
			start_timestamps = [timestamp for timestamp in request.timestamps.start_times]
			end_timestamp = request.timestamps.end_time
		start_times = np.asarray([timestamp.to_sec() for timestamp in start_timestamps])
		end_time = end_timestamp.to_sec()
		
		transforms = self.transform_manager.get_map_transforms(start_times, end_time)

		# Those transforms are from start_time to end_time, expressed in the map frame as the velocity refers to the map frame
		# So we need to rotate those back to the end_time frame
		map_transform = self.get_rotation(self.parameters["node"]["world-frame"], self.parameters["node"]["road-frame"])
		for i in range(transforms.shape[0]):
			transforms[i, :3, 3] = map_transform @ transforms[i, :3, 3]
			transforms[i, :3, :3] = map_transform @ transforms[i, :3, :3] @ map_transform.T

		transform_array = Float64MultiArray()
		transform_array.data = transforms.flatten()
		transform_array.layout.data_offset =  0
		dim = []
		dim.append(MultiArrayDimension("transforms", transforms.shape[0], transforms.shape[0]*transforms.shape[1]*transforms.shape[2]))
		dim.append(MultiArrayDimension("matrix_dim0", transforms.shape[1], transforms.shape[1]*transforms.shape[2]))
		dim.append(MultiArrayDimension("matrix_dim1", transforms.shape[2], transforms.shape[2]))
		transform_array.layout.dim = dim
		return TransformBatchResponse(timestamps=TimeBatch(start_times=start_timestamps, end_time=end_timestamp), transforms=transform_array)
	
	def handle_drop_velocity(self, request):
		if request.unbias:
			end_time = (request.end_time + self.time_discrepancy).to_sec()
		else:
			end_time = request.end_time.to_sec()
		self.transform_manager.drop_velocity(end_time)
		return DropVelocityResponse(done=True)



if __name__ == "__main__":
	np.set_printoptions(threshold=sys.maxsize, suppress=True)
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <parameter-file>")
	else:
		with open(sys.argv[1], "r") as parameterfile:
			parameters = yaml.load(parameterfile, yaml.Loader)

		rospy.init_node(parameters["node"]["transform-node-name"])
		node = TransformBatchServer(parameters)
		rospy.spin()
