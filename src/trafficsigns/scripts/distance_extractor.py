#!/usr/bin/env python3
import sys

import yaml
import cv2 as cv
import numpy as np
import transforms3d.quaternions as quaternions

import rospy
import tf2_ros
import ros_numpy
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

import fish2bird
from circulation.msg import TimeBatch
from circulation.srv import TransformBatch, TransformBatchRequest

from traffic_sign_detection import TrafficSignDetector


DISTANCE_SCALE_MIN = 0
DISTANCE_SCALE_MAX = 160

class DistanceExtractor (object):
	def __init__(self, parameters):

		self.parameters = parameters

		self.image_topic = self.parameters["node"]["image-topic"]
		self.camerainfo_topic = self.parameters["node"]["camerainfo-topic"]
		self.pointcloud_topic = self.parameters["node"]["pointcloud-topic"]
		self.traffic_sign_topic = self.parameters["node"]["traffic-sign-topic"]

		# Initialize the topic subscribers
		self.image_subscriber = rospy.Subscriber(self.image_topic, Image, self.callback_image)
		self.camerainfo_subscriber = rospy.Subscriber(self.camerainfo_topic, CameraInfo, self.callback_camerainfo)
		self.pointcloud_subscriber = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.callback_pointcloud)

		# Initialize the topic publisher
		self.traffic_sign_publisher = rospy.Publisher(self.traffic_sign_topic, String, queue_size=10)
		
		# Initialize the transformation listener
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		# Initialize the traffic sign detector
		self.traffic_sign_detector = None
		self.traffic_sign_detector = TrafficSignDetector()

		# At first everything is null, no image can be produces if one of those is still null
		self.image_frame = None
		self.pointcloud_frame = None
		self.latest_image = None
		self.latest_pointcloud = None
		self.pointcloud_array = []
		self.pointcloud_stamp_array = []
		self.lidar_to_camera = None
		self.distortion_parameters = None
		self.camera_to_image = None
		self.image_stamp = None
		self.pointcloud_stamp = None

		# Initialize the service connections
		rospy.loginfo("Waiting for the TransformBatch service...")
		self.transform_service = None
		rospy.wait_for_service(self.parameters["node"]["transform-service-name"])
		self.transform_service = rospy.ServiceProxy(self.parameters["node"]["transform-service-name"], TransformBatch, persistent=True)

		rospy.loginfo("Everything ready")


	def get_map_transforms(self, start_times, end_time):
		"""Get the transform matrix to transform 3d homogeneous coordinates in the right target timestamp"""
		request = TransformBatchRequest()
		request.timestamps = TimeBatch(start_times=start_times, end_time=end_time)
		request.unbias = False
		tries = 0
		while True:
			try:
				response = self.transform_service(request)
				break
			except rospy.ServiceException as exc:
				if tries > 10:
					rospy.logerr(f"Connection to service {self.parameters['node']['transform-service-name']} failed {tries} times, skipping")
					rospy.logerr(f"Failed with error {exc}")
					raise RuntimeError("Unable to connect to the transform service")
				rospy.logerr(f"Connection to service {self.parameters['node']['transform-service-name']} lost, reconnecting...")
				self.transform_service.close()
				self.transform_service = rospy.ServiceProxy(self.parameters["node"]["transform-service-name"], TransformBatch, persistent=True)
				tries += 1
		transforms = np.asarray(response.transforms.data).reshape(response.transforms.layout.dim[0].size, response.transforms.layout.dim[1].size, response.transforms.layout.dim[2].size)
		start_times_unbiased = response.timestamps.start_times
		end_time_unbiased = response.timestamps.end_time
		return transforms, start_times_unbiased, end_time_unbiased
	

	def update_transforms(self):
		"""Update the lidar-to-camera transform matrix from the tf topic"""
		try:
			# It’s lookup_transform(target_frame, source_frame, …) !!!
			transform = self.tf_buffer.lookup_transform(self.image_frame, self.pointcloud_frame, rospy.Time(0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			return

		# Build the matrix elements
		rotation_message = transform.transform.rotation
		rotation_quaternion = np.asarray((rotation_message.w, rotation_message.x, rotation_message.y, rotation_message.z))
		rotation_matrix = quaternions.quat2mat(rotation_quaternion)
		translation_message = transform.transform.translation
		translation_vector = np.asarray((translation_message.x, translation_message.y, translation_message.z)).reshape(3, 1)
		
		# Build the complete transform matrix
		self.lidar_to_camera = np.concatenate((
			np.concatenate((rotation_matrix, translation_vector), axis=1),
			np.asarray((0, 0, 0, 1)).reshape((1, 4))
		), axis=0)


	def callback_image(self, data):
		"""Extract an image from the camera"""
		# rospy.loginfo("Received an image")
		self.image_frame = data.header.frame_id
		self.image_stamp = data.header.stamp
		self.latest_image = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, 3))
		if self.image_stamp >= self.pointcloud_stamp_array[0]:
			self.convert_pointcloud()


	def callback_pointcloud(self, data):
		"""Extract a point cloud from the lidar"""
		# rospy.loginfo("Received a point cloud")
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
			self.camera_to_image is None or self.traffic_sign_detector is None):
			return

		self.update_transforms()

		pointcloud = self.pointcloud_array[0]
		pointcloud_stamp = self.pointcloud_stamp_array[0]
		img = self.latest_image
		img_stamp = self.image_stamp

		self.pointcloud_array = []
		self.pointcloud_stamp_array = []


		# We can add the 3 lines below to transform the lidar points coordinates in the image timestamp as target
		# transforms, _, _ = self.get_map_transforms([pointcloud_stamp], img_stamp)
		# transform = transforms[0]
		# pointcloud = transform @ pointcloud

		lidar_coordinates_in_image = self.lidar_to_image(pointcloud)

		camera_pointcloud = self.lidar_to_camera @ pointcloud

		# Calculate the distance to each lidar point
		distances = np.linalg.norm(pointcloud, axis=0)

		# Get the annotated image and detected traffic signs labels and coordinates
		img, traffic_signs = self.traffic_sign_detector.get_traffic_signs(img)

		# If at least one traffic sign is detected
		if len(traffic_signs) >= 0:

			traffic_sign_distances = np.zeros((len(traffic_signs), 1))
			nb_distances = np.zeros((len(traffic_signs), 1))

			# Write all points to the final image
			i=0
			for distance, point,  in zip(distances, lidar_coordinates_in_image.T):
				# Filter out points that are not in the image dimension or behind the camera
				if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0] and camera_pointcloud[2,i]>=0:
					cv.circle(img, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
					j=0
					for traffic_sign in traffic_signs:
						label, x, y, w, h = traffic_sign
						# Filter out points that are not in the traffic sign bounding box
						if x <= point[0] < x+w and y <= point[1] < y+h:
							nb_distances[j] += 1
							traffic_sign_distances[j] += distance
						j+=1
				i+=1

			# Computes the mean of all distances (lidar points projected in the bounding box) from each traffic sign
			traffic_sign_distances = traffic_sign_distances / nb_distances

			# Publish on topic and display distances of each traffic sign
			for traffic_sign, distance in zip(traffic_signs, traffic_sign_distances):
				label, x, y, w, h = traffic_sign

				distance = round(distance[0],2)

				img = cv.putText(img, 'd = '+str(distance)+' m', (x, y-25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

				if 30 > distance > 0:
					string_to_publish = label + ';' + str(distance)
					self.traffic_sign_publisher.publish(string_to_publish)

				print(label)
		
		img = cv.cvtColor(self.latest_image, cv.COLOR_BGR2RGB)
		cv.imshow('image', img)

		cv.waitKey(5)


if __name__ == "__main__":

	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <parameter-file>")
	else:
		with open(sys.argv[1], "r") as parameterfile:
			parameters = yaml.load(parameterfile, yaml.Loader)
		rospy.init_node("traffic_sign_distances")
		node = DistanceExtractor(parameters)
		rospy.spin()
		cv.destroyAllWindows()

