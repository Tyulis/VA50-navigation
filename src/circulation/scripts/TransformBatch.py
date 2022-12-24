#!/usr/bin/env python3

import sys
import yaml
import numpy as np

import rospy
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
		
		self.transform_manager = positioning.TransformManager()
		if self.parameters["node"]["time-discrepancy"]:
			self.time_discrepancy_buffer = []
			self.time_discrepancy = None
		else:
			self.time_discrepancy = rospy.Duration(0)

		self.velocity_subscriber = rospy.Subscriber(self.parameters["node"]["velocity-topic"], TwistStamped, self.callback_velocity, queue_size=10)
		if self.time_discrepancy is None:
			rospy.loginfo("Service initialized, waiting for time discrepancy estimation")
		else:
			rospy.loginfo("Ready")
	
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
