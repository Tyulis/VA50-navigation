#!/usr/bin/env python3

#   Copyright 2023 Grégori MIGNEROT, Élian BELMONTE, Benjamin STACH
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import sys
import cProfile
from threading import Lock

import yaml
import cv2 as cv
import numpy as np
import transforms3d.quaternions as quaternions
from sklearn.neighbors import KernelDensity

import rospy
import tf2_ros
import ros_numpy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

import fish2bird
from trafficsigns.msg import TrafficSign, TrafficSignStatus
from transformtrack.srv import TransformBatch, TransformBatchRequest

from traffic_sign_detection import TrafficSignDetector
from traffic_light_detection import detect_traffic_lights


DISTANCE_SCALE_MIN = 0
DISTANCE_SCALE_MAX = 160

class DistanceExtractor (object):
	def __init__(self, parameters, no_lights, no_signs):
		self.parameters = parameters
		self.no_lights = no_lights
		self.no_signs = no_signs

		self.image_topic = self.parameters["node"]["image-topic"]
		self.camerainfo_topic = self.parameters["node"]["camerainfo-topic"]
		self.pointcloud_topic = self.parameters["node"]["pointcloud-topic"]
		self.traffic_sign_topic = self.parameters["node"]["traffic-sign-topic"]
		self.visualization_topic = self.parameters["node"]["trafficsigns-viz-topic"]

		# Initialize the topic publisher
		self.traffic_sign_publisher = rospy.Publisher(self.traffic_sign_topic, TrafficSignStatus, queue_size=10)
		self.status_seq = 0
		
		# Initialize the transformation listener
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		# Initialize the traffic sign detector
		self.traffic_sign_detector = None
		if not self.no_signs:
			self.traffic_sign_detector = TrafficSignDetector()

		# At first everything is null, no image can be produces if one of those is still null
		self.image_frame = None
		self.pointcloud_frame = None
		self.latest_image = None
		self.latest_pointcloud = None
		self.pointcloud_array = []
		self.pointcloud_stamp_array = []
		self.lidar_to_camera = None
		self.lidar_to_baselink = None
		self.distortion_parameters = None
		self.camera_to_image = None
		self.image_stamp = None
		self.pointcloud_stamp = None

		rospy.loginfo("Waiting for the TransformBatch service...")
		self.transform_service = None
		rospy.wait_for_service(self.parameters["node"]["transform-service-name"])
		self.transform_service = rospy.ServiceProxy(self.parameters["node"]["transform-service-name"], TransformBatch, persistent=True)
		self.transform_service_lock = Lock()

		# Initialize the topic subscribers
		self.image_subscriber = rospy.Subscriber(self.image_topic, Image, self.callback_image, queue_size=1, buff_size=2**28)
		self.camerainfo_subscriber = rospy.Subscriber(self.camerainfo_topic, CameraInfo, self.callback_camerainfo)
		self.pointcloud_subscriber = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.callback_pointcloud)
		self.visualization_publisher = rospy.Publisher(self.visualization_topic, Image, queue_size=10)

		rospy.loginfo("Everything ready")	

	def get_transform(self, source_frame, target_frame):
		"""Update the lidar-to-camera transform matrix from the tf topic"""
		try:
			# It’s lookup_transform(target_frame, source_frame, …) !!!
			transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			return

		# Build the matrix elements
		rotation_message = transform.transform.rotation
		rotation_quaternion = np.asarray((rotation_message.w, rotation_message.x, rotation_message.y, rotation_message.z))
		rotation_matrix = quaternions.quat2mat(rotation_quaternion)
		translation_message = transform.transform.translation
		translation_vector = np.asarray((translation_message.x, translation_message.y, translation_message.z)).reshape(3, 1)
		
		# Build the complete transform matrix
		return np.concatenate((
			np.concatenate((rotation_matrix, translation_vector), axis=1),
			np.asarray((0, 0, 0, 1)).reshape((1, 4))
		), axis=0)

	def get_map_transforms(self, start_times, end_time):
		"""Get a batch of transforms of the vehicle frame from the given `start_times` to `end_time`
		   - start_times : list<rospy.Time> : Timestamps to get the transforms from
		   - end_time    : rospy.Time       : Target timestamp, to get the transforms to
		<----------------- ndarray[N, 4, 4] : 3D homogeneous transform matrices to transform points in the vehicle frame
		                                      at `start_times[i]` to the vehicle frame at `end_time`
		<----------------- list<rospy.Time> : Unbiased start times. This is an artifact from the time when the simulator gave incoherent timestamps,
		                                      same as `start_times` on the latest versions
		<----------------- rospy.Time       : Unbiased end time, same as end_time on the latest versions of the simulator
		"""
		# Build the request to the transform service, see srv/TransformBatch.srv and msg/TimeBatch.msg for info
		request = TransformBatchRequest()
		request.start_times = start_times
		request.end_time = end_time

		# Now send the request and get the response
		# We use persistent connections to improve efficiency, and ROS advises to implement some reconnection logic
		# in case the network gets in the way, so in case of disconnection, retry 10 times to reconnect then fail
		tries = 0
		while True:
			try:
				# Apparently, when a call to the service is pending, the node is free to service other callbacks,
				# including callback_trafficsign that also call this service
				# So with the traffic signs subscriber active, it’s only a matter of time until both get to their transform service call concurrently
				# For some reason, ROS allows it, and for some reason it deadlocks ROS as a whole
				# So let’s throw in a lock to prevent ROS from killing itself
				with self.transform_service_lock:
					response = self.transform_service(request)
				break
			except rospy.ServiceException as exc:
				if tries > 10:
					rospy.logerr(f"Connection to service {self.parameters['node']['transform-service-name']} failed {tries} times, skipping")
					rospy.logerr(f"Failed with error : {exc}")
					raise RuntimeError("Unable to connect to the transform service")
				rospy.logerr(f"Connection to service {self.parameters['node']['transform-service-name']} lost, reconnecting...")
				self.transform_service.close()
				self.transform_service = rospy.ServiceProxy(self.parameters["node"]["transform-service-name"], TransformBatch, persistent=True)
				tries += 1
		
		# The call was successful, get the transforms in the right format and return
		# The transpose is because the individual matrices are transmitted in column-major order
		transforms = np.asarray(response.transforms.data).reshape(response.transforms.layout.dim[0].size, response.transforms.layout.dim[1].size, response.transforms.layout.dim[2].size).transpose(0, 2, 1)
		distances = np.asarray(response.distances)
		return transforms, distances


	def callback_image(self, data):
		"""Extract an image from the camera"""
		self.image_frame = data.header.frame_id
		self.image_stamp = data.header.stamp
		self.latest_image = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, 3))
		if self.image_stamp >= self.pointcloud_stamp_array[0]:
			self.convert_pointcloud()
			#cProfile.runctx("self.convert_pointcloud()", globals(), locals())


	def callback_pointcloud(self, data):
		"""Extract a point cloud from the lidar"""
		self.pointcloud_frame = data.header.frame_id
		self.pointcloud_stamp = data.header.stamp

		# Extract the (x, y, z) points from the PointCloud2 message
		points_3d = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data, remove_nans=True)

		# Convert to a matrix with points as columns, in homogeneous coordinates
		self.latest_pointcloud = np.concatenate((
			points_3d.transpose(),
			np.ones(points_3d.shape[0]).reshape((1, points_3d.shape[0]))
		), axis=0)

		self.pointcloud_stamp_array.append(self.pointcloud_stamp)
		self.pointcloud_array.append(self.latest_pointcloud)


	def callback_camerainfo(self, data):
		"""Callback called when a new camera info message is published"""
		# fish2bird only supports the camera model defined by Christopher Mei
		if data.distortion_model.lower() != "mei":
			rospy.logerr(f"Bad distortion model : {data.distortion_model}")
			return
		self.camera_to_image = np.asarray(data.P).reshape((3, 4))
		self.distortion_parameters = data.D


	def lidar_to_image(self, pointcloud):
		return fish2bird.target_to_image(pointcloud, self.lidar_to_camera, self.camera_to_image, self.distortion_parameters[0])

	def convert_pointcloud(self):
		"""Superimpose a point cloud from the lidar onto an image from the camera and publish the distance to the traffic sign bounding box on the topic"""
		# If some info is missing, can’t output an image
		if (self.image_frame is None or self.pointcloud_frame is None or
			self.latest_image is None or self.latest_pointcloud is None or
			self.camera_to_image is None):  # or self.traffic_sign_detector is None):
			return

		self.lidar_to_camera = self.get_transform(self.pointcloud_frame, self.image_frame)
		self.lidar_to_baselink = self.get_transform(self.pointcloud_frame, self.parameters["node"]["road-frame"])

		pointcloud = np.ascontiguousarray(self.pointcloud_array[0])
		pointcloud_stamp = self.pointcloud_stamp_array[0]
		img = self.latest_image
		img_stamp = self.image_stamp

		transforms, distances = self.get_map_transforms([pointcloud_stamp], img_stamp)
		pointcloud = transforms[0] @ pointcloud

		self.pointcloud_array = []
		self.pointcloud_stamp_array = []

		lidar_coordinates_in_image = self.lidar_to_image(pointcloud)

		camera_pointcloud = self.lidar_to_camera @ pointcloud

		# Get the annotated image and detected traffic signs labels and coordinates
		traffic_signs = []
		if not self.no_signs:
			img, signs = self.traffic_sign_detector.get_traffic_signs(img)
			traffic_signs.extend(signs)
		if not self.no_lights:
			img, traffic_lights = detect_traffic_lights(img)
			traffic_signs.extend(traffic_lights)

		# Visualize the lidar data projection onto the image
		for i, point in enumerate(lidar_coordinates_in_image.T):
				# Filter out points that are not in the image dimension or behind the camera
				if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0] and camera_pointcloud[2, i] >=0:
					cv.drawMarker(img, (int(point[0]), int(point[1])), (0, 255, 0), cv.MARKER_CROSS, 4)

		# If at least one traffic sign is detected
		if len(traffic_signs) > 0:
			message = TrafficSignStatus()
			message.header = Header(seq=self.status_seq, stamp=img_stamp, frame_id=self.parameters["node"]["road-frame"])
			self.status_seq += 1
			sign_messages = []

			for sign in traffic_signs:
				result = TrafficSign()
				result.category = sign.category
				result.type = sign.type
				result.confidence = sign.confidence

				relevant_points_filter = ((sign.x <= lidar_coordinates_in_image[0]) & (lidar_coordinates_in_image[0] <= sign.x + sign.width) &
		    					          (sign.y <= lidar_coordinates_in_image[1]) & (lidar_coordinates_in_image[1] <= sign.y + sign.height))
				relevant_points = pointcloud[:, relevant_points_filter]
				
				# We can still publish that we’ve seen it just in case, but we have no information on its position whatsoever
				if relevant_points.shape[1] == 0:
					result.x = np.nan
					result.y = np.nan
					result.z = np.nan
				else:
					# Maximum density estimation to disregard the points that might be in the hitbox but physically behind the sign
					baselink_points = self.lidar_to_baselink @ relevant_points
					density_model = KernelDensity(kernel="epanechnikov", bandwidth=np.linalg.norm([sign.width, sign.height]) / 2)
					density_model.fit(baselink_points.T)
					point_density = density_model.score_samples(baselink_points.T)
					position_estimate = baselink_points[:, np.argmax(point_density)]

					result.x = position_estimate[0]
					result.y = position_estimate[1]
					result.z = position_estimate[2]
				
					img = cv.putText(img, f'd = {np.linalg.norm(position_estimate)} m', (sign.x, sign.y-25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
				
				sign_messages.append(result)
			message.traffic_signs = sign_messages
			self.traffic_sign_publisher.publish(message)
		
		#image_message = Image(height=img.shape[0], width=img.shape[1], data=tuple(img.flatten()))
		#self.visualization_publisher.publish(image_message)
		img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
		cv.imshow('Panneaux', img)
		cv.waitKey(5)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <parameter-file> [--no-lights] [--no-signs]")
	else:
		with open(sys.argv[1], "r") as parameterfile:
			parameters = yaml.load(parameterfile, yaml.Loader)
		rospy.init_node("traffic_sign_distances")
		node = DistanceExtractor(parameters, "--no-lights" in sys.argv, "--no-signs" in sys.argv)
		rospy.spin()

