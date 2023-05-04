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

"""
Main module of the ROS node that builds the vehicle’s trajectory

NOTES :
    - There is no static reference frame allowed here, so our reference frame is the vehicle (base_link)
	  As such, it is very important to keep track of the time at which each measurement is taken, whatever it is,
	  so that we know how to pull it back to the local frame at the relevant moment
	- All discrete curves in this node and its submodules are using COLUMN VECTORS (numpy shape [2, N])

TROUBLESHOOTING :
    - If the visualization window often stays ridiculously small, uncomment the marked lines in the method `TrajectoryVisualizer.update`
"""

# ═══════════════════════════ BUILT-IN IMPORTS ════════════════════════════ #
import sys
import time
import enum
import cProfile
from threading import Lock
from collections import Counter

# ══════════════════════════ THIRD-PARTY IMPORTS ══════════════════════════ #
import yaml
import cv2 as cv
import numpy as np
import transforms3d.quaternions as quaternions
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN

# ══════════════════════════════ ROS IMPORTS ══════════════════════════════ #
import rospy
import tf2_ros
from std_msgs.msg import UInt8, Float64MultiArray, MultiArrayDimension, Header
from sensor_msgs.msg import Image, CameraInfo
from transformtrack.srv import TransformBatch, TransformBatchRequest, DropVelocity, DropVelocityRequest
from transformtrack.msg import TimeBatch
from circulation.msg import Trajectory
from trafficsigns.msg import TrafficSignStatus, TrafficSign
from visualization.msg import VizUpdate

# ═══════════════════════ CYTHON EXTENSION MODULES ════════════════════════ #
import linetrack
import trajeometry
import fish2bird
import fuzzylines
import trajectorybuild

# TODO : More resilient lane detection
# TODO : Autonomous intersection detection
# TODO : No direction panic mode
# TODO : Ensure a better base point continuity in trajectories


INTERSECTION_SIGNS = ( 
    'yield', 
    'stop', 
    'right-only', 
    'left-only', 
    'ahead-only', 
    'straight-right-only', 
    'straight-left-only', 
    'keep-right', 
    'keep-left',
)


class NavigationMode (enum.Enum):
	CRUISE = 0
	FORWARD_SKIP = 110
	LEFT_TURN = 111
	RIGHT_TURN = 112
	PANIC_CORE_BREACH = 500
	PANIC_UNSUPPORTED = 501
	PANIC_EXCEPTION = 502
	PANIC_NO_DIRECTION = 510

class Direction:
	DEAD_END = 0b0000
	FORWARD = 0b0001
	LEFT = 0b0010
	RIGHT = 0b0100

# class TrajectoryVisualizer (object):
# 	"""Quick-and-dirty visualization window management
# 	   There are 2 visualizations, just merge them into one and call cv.imshow"""
	
# 	def __init__(self, parameters):
# 		self.parameters = parameters
# 		self.publisher = rospy.Publisher(self.parameters["node"]["visualization-topic"], VizUpdate, queue_size=10)

# 	def update_line_detection(self, be_binary, lines, left_line_index, right_line_index, markings):
# 		"""Generate and update the left visualization from the preprocessed image and the detected lines and markings
# 		   - be_binary        : ndarray[y, x]       : Preprocessed camera image (binary edge-detected bird-eye view)
# 		   - lines            : list<ndarray[2, N]> : Detected discrete curves in the image
# 		   - left_line_index  : int                 : Index of the left lane marking in the `lines` list, or None
# 		   - right_line_index : int                 : Index of the right lane marking in the `lines` list, or None
# 		   - markings         : dict<str, …>        : Dictionary of detected road markings. Currently only supports `crosswalks`
# 		"""
# 		line_viz = cv.merge((be_binary, be_binary, be_binary))
# 		for line in lines:
# 			cv.polylines(line_viz, [line.astype(int).transpose()], False, (0, 200, 0), 2)
# 		if left_line_index is not None:
# 			cv.polylines(line_viz, [lines[left_line_index].astype(int).transpose()], False, (255, 0, 0), 4)
# 		if right_line_index is not None:
# 			cv.polylines(line_viz, [lines[right_line_index].astype(int).transpose()], False, (0, 100, 255), 4)

# 		for i, crosswalk in enumerate(markings["crosswalks"]):
# 			color = ((int(i * 255 / len(markings["crosswalks"])) + 30) % 255, 255, 255)
# 			for rectangle in crosswalk:
# 				cv.fillPoly(line_viz, [rectangle.astype(int).transpose()], color)

# 		self.update(line_viz, self.parameters["visualization"]["circulation-lines-id"])

# 	def update_trajectory_construction(self, viz):
# 		"""Update the right visualization with the given image"""
# 		self.update(viz, self.parameters["visualization"]["circulation-trajectory-id"])

# 	def update(self, image, id):
# 		"""Update the visualization window"""
# 		image_message = Image(height=image.shape[0], width=image.shape[1], data=tuple(image.flatten()))
# 		message = VizUpdate(id=id, image=image_message)
# 		self.publisher.publish(message)
	

class TrajectoryVisualizer (object):
	"""Quick-and-dirty visualization window management
	   There are 2 visualizations, just merge them into one and call cv.imshow"""
	
	def __init__(self, parameters):
		self.line_viz = None
		self.trajectory_viz = None

	def update_line_detection(self, be_binary, lines, left_line_index, right_line_index, markings):
		"""Generate and update the left visualization from the preprocessed image and the detected lines and markings
		   - be_binary        : ndarray[y, x]       : Preprocessed camera image (binary edge-detected bird-eye view)
		   - lines            : list<ndarray[2, N]> : Detected discrete curves in the image
		   - left_line_index  : int                 : Index of the left lane marking in the `lines` list, or None
		   - right_line_index : int                 : Index of the right lane marking in the `lines` list, or None
		   - markings         : dict<str, …>        : Dictionary of detected road markings. Currently only supports `crosswalks`
		"""
		self.line_viz = cv.merge((be_binary, be_binary, be_binary))
		for line in lines:
			cv.polylines(self.line_viz, [line.astype(int).transpose()], False, (0, 200, 0), 2)
		if left_line_index is not None:
			cv.polylines(self.line_viz, [lines[left_line_index].astype(int).transpose()], False, (255, 0, 0), 4)
		if right_line_index is not None:
			cv.polylines(self.line_viz, [lines[right_line_index].astype(int).transpose()], False, (0, 100, 255), 4)

		for i, crosswalk in enumerate(markings["crosswalks"]):
			color = ((int(i * 255 / len(markings["crosswalks"])) + 30) % 255, 255, 255)
			for rectangle in crosswalk:
				cv.fillPoly(self.line_viz, [rectangle.astype(int).transpose()], color)

		self.update()

	def update_trajectory_construction(self, viz):
		"""Update the right visualization with the given image"""
		self.trajectory_viz = viz
		self.update()

	def update(self):
		"""Update the visualization window"""
		if self.line_viz is None or self.trajectory_viz is None:
			return
		# Just merge both images
		full_viz = cv.cvtColor(np.concatenate((self.line_viz, 255*np.ones((self.line_viz.shape[0], 30, 3), dtype=np.uint8), self.trajectory_viz), axis=1), cv.COLOR_RGB2BGR)
		
		# Uncomment these if your visualization window often stays tiny
		## cv.namedWindow("viz", cv.WINDOW_NORMAL)
		## cv.resizeWindow("viz", full_viz.shape[1], full_viz.shape[0])
		cv.imshow("viz", full_viz)
		cv.waitKey(1)

class IntersectionHint (object):
	"""Hold the position and informations about anything that may indicate an intersection"""

	def __init__(self, category, type, position, timestamp, confidence):
		self.category = category
		self.type = type
		self.positions = [position]
		self.position_timestamps = [timestamp]
		self.confidences = [confidence]
	
	def merge(self, hint):
		self.positions.extend(hint.positions)
		self.position_timestamps.extend(hint.position_timestamp)
		self.confidences.extend(hint.confidences)
	
	def confidence(self):
		return 1 - np.sqrt(np.sum((1 - self.confidences)**2)) / len(self.confidences)

	def direction_hint(self):
		if self.category != "trafficsign":
			return Direction.FORWARD | Direction.LEFT | Direction.RIGHT

		if self.type in ("right-only", "keep-right"):
			return Direction.RIGHT
		elif self.type in ("left-only", "keep-left"):
			return Direction.LEFT
		elif self.type == "ahead-only":
			return Direction.FORWARD
		elif self.type == "straight-left-only":
			return Direction.FORWARD | Direction.LEFT
		elif self.type == "straight-right-only":
			return Direction.FORWARD | Direction.RIGHT
		else:
			return Direction.FORWARD | Direction.LEFT | Direction.RIGHT

	def __hash__(self):
		return id(self)
		


class TrajectoryExtractorNode (object):
	"""Main class for the ROS node that manages the trajectory"""

	#                        ╔══════════════════════╗                       #
	# ═══════════════════════╣    INITIALISATION    ╠══════════════════════ #
	#                        ╚══════════════════════╝                       #

	def __init__(self, parameters):
		"""Initialize the node and everything that it needs
		   - parameters   : dict<str: …>        : Node parameters, from the parameter file
		"""
		self.parameters = parameters

		# Trajectory history buffers
		self.trajectory_buffer = []                   # History of trajectories estimated from individual frames
		self.trajectory_scores = []                   # Scores associated to each trajectory in `trajectory_buffer`
		self.trajectory_timestamps = []               # Timestamp each trajectory in `trajectory_buffer` corresponds to
		self.current_trajectory = None                # Current trajectory, that has been last published or will be next published
		self.current_trajectory_timestamp = None      # Timestamp the `current_trajectory` corresponds to
		self.navigation_mode = NavigationMode.CRUISE  # Current navigation mode (cruise, intersection, …)

		# Just stats
		self.time_buffer = []  # History to make performance statistics

		# Initialize the fuzzy systems
		self.init_fuzzysystems()

		# Initialize the visualization
		if self.parameters["node"]["visualize"]:
			self.visualisation = TrajectoryVisualizer(self.parameters)
		else:
			self.visualisation = None

		# Initialize the transformation listener
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		# Bird-eye projection parameters, for convenience
		self.birdeye_range_x = (-self.parameters["birdeye"]["x-range"], self.parameters["birdeye"]["x-range"])
		self.birdeye_range_y = (self.parameters["birdeye"]["roi-y"], self.parameters["birdeye"]["y-range"])

		# Intersection management
		self.next_direction = Direction.FORWARD
		self.intersection_hints = []
		self.last_lane_rejoin = None

		# Initialize the service connections
		rospy.loginfo("Waiting for the TransformBatch service...")
		self.transform_service = None
		self.drop_service = None
		rospy.wait_for_service(self.parameters["node"]["transform-service-name"])
		rospy.wait_for_service(self.parameters["node"]["drop-service-name"])
		self.transform_service = rospy.ServiceProxy(self.parameters["node"]["transform-service-name"], TransformBatch, persistent=True)
		self.drop_service = rospy.ServiceProxy(self.parameters["node"]["drop-service-name"], DropVelocity, persistent=True)
		self.transform_service_lock = Lock()

		# Initialize the topic subscribers (last to avoid too early messages while other things are not yet initialized)
		self.image_subscriber = rospy.Subscriber(self.parameters["node"]["image-topic"], Image, self.callback_image, queue_size=1, buff_size=2**28)
		self.camerainfo_subscriber = rospy.Subscriber(self.parameters["node"]["camerainfo-topic"], CameraInfo, self.callback_camerainfo, queue_size=1)
		self.direction_subscriber = rospy.Subscriber(self.parameters["node"]["direction-topic"], UInt8, self.callback_direction)
		self.trafficsign_subscriber = rospy.Subscriber(self.parameters["node"]["traffic-sign-topic"], TrafficSignStatus, self.callback_trafficsign)
		self.trajectory_publisher = rospy.Publisher(self.parameters["node"]["trajectory-topic"], Trajectory, queue_size=10)
		self.trajectory_seq = 0  # Sequential number of published trajectories

		rospy.loginfo("Ready")

	def init_fuzzysystems(self):
		"""Initialize the fuzzy systems used by the lane detection"""
		line_variables = ("forward-distance", "line-distance", "line-lengths", "parallel-distances", "parallel-angles")
		line_centers = np.asarray([self.parameters["fuzzy-lines"]["centers"][variable] for variable in line_variables])
		line_malus = np.asarray([self.parameters["fuzzy-lines"]["malus"][variable] for variable in line_variables], dtype=int)
		line_output_centers = np.asarray(self.parameters["fuzzy-lines"]["centers"]["output"])
		self.lane_system = fuzzylines.FuzzySystem(line_centers, line_malus, line_output_centers, self.parameters["fuzzy-lines"]["base-score"])

	#                        ╔══════════════════════╗                       #
	# ═══════════════════════╣ SUBSCRIBER CALLBACKS ╠══════════════════════ #
	#                        ╚══════════════════════╝                       #

	def callback_image(self, message):
		"""Callback called when an image is published from the camera
		   - message : sensor_msgs.msg.Image : Message from the camera
		"""
		# No transform service : cannot proceed
		if self.transform_service is None or self.drop_service is None:
			return
		
		if self.is_panic():
			return
		
		# Extract the image and the timestamp at which it was taken, critical for synchronisation
		rospy.loginfo("------ Received an image")
		image = np.frombuffer(message.data, dtype=np.uint8).reshape((message.height, message.width, 3))
		self.compute_trajectory(image, message.header.stamp, message.header.frame_id)
		#cProfile.runctx("self.compute_trajectory(image, message.header.stamp, message.header.frame_id)", globals(), locals())

	def callback_camerainfo(self, message):
		"""Callback called when a new camera info message is published
		   - message : sensor_msgs.msg.CameraInfo : Message with metadata about the camera
		"""
		# fish2bird only supports the camera model defined by Christopher Mei
		if message.distortion_model.lower() != "mei":
			rospy.logerr(f"Bad distortion model : {message.distortion_model}")
			return
		self.camera_to_image = np.asarray(message.P).reshape((3, 4))[:, :3]
		self.distortion_parameters = message.D
	
	def callback_direction(self, message):
		"""Callback called when a direction is sent from the navigation nodes
		   - message : std_msgs.msg.Uint8 : Message with the direction (same values as Direction.FORWARD, .LEFT and .RIGHT)
		"""
		if message.data == Direction.FORWARD:
			rospy.loginfo("Updated next direction to FORWARD")
		elif message.data == Direction.LEFT:
			rospy.loginfo("Updated next direction to LEFT")
		elif message.data == Direction.RIGHT:
			rospy.loginfo("Updated next direction to RIGHT")
		else:
			rospy.logerr(f"Invalid direction ID received : {message.data}")
			return
		self.next_direction = message.data
	
	def callback_trafficsign(self, message):
		"""Callback called when traffic signs are detected and received
		   - message : trafficsigns.msg.TrafficSignStatus : Message with the detected traffic signs data
		"""
		for trafficsign in message.traffic_signs:
			if trafficsign.type in INTERSECTION_SIGNS and trafficsign.confidence > 0.6:
				self.add_intersection_hint(IntersectionHint("trafficsign", trafficsign.type, (trafficsign.x, trafficsign.y, trafficsign.z), message.header.stamp, trafficsign.confidence))

	#               ╔═══════════════════════════════════════╗               #
	# ══════════════╣ TRANSFORM MANAGEMENT AND MEASUREMENTS ╠══════════════ #
	#               ╚═══════════════════════════════════════╝               #

	def get_transform(self, source_frame, target_frame):
		"""Get the latest transform matrix from `source_frame` to `target_frame`
		   - source_frame : str           : Name of the source frame
		   - target_frame : str           : Name of the target frame
		<---------------- : ndarray[4, 4] : 3D homogeneous transform matrix to convert from `source_frame` to `target_frame`,
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
		request.timestamps = TimeBatch(start_times=start_times, end_time=end_time)
		request.unbias = True

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
		start_times_unbiased = start_times  # response.timestamps.start_times
		end_time_unbiased = end_time  # response.timestamps.end_time
		return transforms, start_times_unbiased, end_time_unbiased
	
	def drop_velocity(self, end_time):
		"""Call the DropVelocity service, such that the TransformBatch service discards its old velocity data
		   and doesn’t unnecessarily clutter its memory and performance
		   - end_time : rospy.Time : Discard all velocity data prior to this timestamp
		"""
		request = DropVelocityRequest(end_time=end_time, unbias=True)
		
		# Same reconnection logic as .get_map_transforms
		tries = 0
		while True:
			try:
				response = self.drop_service(request)
				break
			except rospy.ServiceException as exc:
				if tries > 10:
					rospy.logerr(f"Connection to service {self.parameters['node']['drop-service-name']} failed {tries} times, skipping")
					rospy.logerr(f"Failed with error : {exc}")
					raise RuntimeError("Unable to connect to the velocity drop service")
				rospy.logerr(f"Connection to service {self.parameters['node']['drop-service-name']} lost, reconnecting...")
				self.drop_service.close()
				self.drop_service = rospy.ServiceProxy(self.parameters["node"]["drop-service-name"], DropVelocity, persistent=True)
				tries += 1
	
	#                 ╔═══════════════════════════════════╗                 #
	# ════════════════╣ NAVIGATION MODE SWITCH PROCEDURES ╠════════════════ #
	#                 ╚═══════════════════════════════════╝                 #

	def match_intersection_hint(self, existing_hint, hint):
		if hint.category != existing_hint.category:
			return False
		elif hint.type != existing_hint.type:
			return False
		
		transforms, _, _ = self.get_map_transforms(existing_hint.position_timestamps, hint.position_timestamps[-1])
		existing_positions = np.asarray([transform @ np.concatenate((position, [1])).reshape(-1, 1) for transform, position in zip(transforms, existing_hint.positions)])[:2]
		existing_centroid = np.mean(existing_positions, axis=1)
		hint_position = np.asarray(hint.positions[-1])
		return np.linalg.norm(existing_centroid - hint_position) < self.parameters["intersection"]["intersection-hint-match-threshold"][hint.category]

	def add_intersection_hint(self, hint):
		# Skip hints found too close to the last intersection
		if self.navigation_mode != NavigationMode.CRUISE:
			return
		
		if self.last_lane_rejoin is not None:
			transforms, _, _ = self.get_map_transforms([self.last_lane_rejoin], hint.position_timestamps[-1])
			distance = np.linalg.norm(transforms[0][:3, 3])
			print(f"Distance since rejoin : {distance}")
			if distance < self.parameters["intersection"]["hint-detection-buffer"]:
				return
			else:
				self.last_lane_rejoin = None
		
		for existing_hint in self.intersection_hints:
			if self.match_intersection_hint(existing_hint, hint):
				existing_hint.merge(hint)
				break
		else:
			self.intersection_hints.append(hint)
	
	def distance_till_rejoin(self, image_timestamp):
		"""In intersection navigation modes, the vehicle follows a predetermined trajectory,
		   then tries to catch the new lane to follow after some distance stored in `self.rejoin_distance`
		   Check the distance remaining until that distance is reached
		   - image_timestamp : rospy.Time : Timestamp to measure the distance at
		<--------------------- float      : Signed distance until the rejoin distance
		                                    (negative if the vehicle is already farther than `self.rejoin_distance`)"""
		transform = self.get_map_transforms([self.current_trajectory_timestamp], image_timestamp)[0][0]
		distance = np.linalg.norm(transform[:3, 3])  # Compute the distance from the translation vector in the transform matrix
		return self.rejoin_distance - distance
	
	def next_intersection(self, image_timestamp):
		if len(self.intersection_hints) == 0:
			return None, None

		hint_indices = []
		hint_positions = []
		hint_timestamps = []
		for i, hint in enumerate(self.intersection_hints):
			hint_positions.extend(hint.positions)
			hint_indices.extend([i] * len(hint.positions))
			hint_timestamps.extend(hint.position_timestamps)
		
		if len(hint_positions) == 0:
			return None, None
		
		transforms, _, _ = self.get_map_transforms(hint_timestamps, image_timestamp)

		# Project everything onto the current directional vector (0, 1, 0) in local coordinates
		current_distances = np.hstack([transform @ np.concatenate((position, [1])).reshape(-1, 1) for transform, position in zip(transforms, hint_positions)])[1]
		
		# Disregard points that are behind the vehicle
		forward_filter = np.nonzero(current_distances >= 0)[0]
		current_distances = current_distances[forward_filter]
		hint_indices = [index for i, index in enumerate(hint_indices) if i in forward_filter]
		
		if len(current_distances) <= 2:
			selected_index = np.argmin(current_distances)
			hint = self.intersection_hints[hint_indices[selected_index]]
			return current_distances[selected_index], hint.direction_hint()
		
		labels = DBSCAN(eps=2, min_samples=2).fit_predict(current_distances.reshape(-1, 1))
		if np.max(labels) < 0:
			selected_index = np.argmin(current_distances)
			hint = self.intersection_hints[hint_indices[selected_index]]
			return current_distances[selected_index], hint.direction_hint()
		
		cluster_distances = [np.mean(current_distances[labels == label]) for label in np.sort(np.unique(labels)) if label >= 0]
		selected_cluster = np.argmin(cluster_distances)
		hints = {self.intersection_hints[hint_indices[index]] for index in np.nonzero(labels == selected_cluster)[0]}
		direction_hint = Direction.FORWARD | Direction.LEFT | Direction.RIGHT
		for hint in hints:
			direction_hint &= hint.direction_hint()
		return cluster_distances[selected_cluster], direction_hint
		
	
	def update_intersection(self, image_timestamp):
		"""Take a measurement and update the next intersection estimate based on all the measurements in the history buffers,
		   and act accordingly (switch navigation mode, ...)
		   - image_timestamp : rospy.Time : Timestamp to estimate the informations about
		"""
		# Don’t estimate the next intersection if the vehicle is already on an intersection
		if self.navigation_mode != NavigationMode.CRUISE:
			return

		intersection_distance, intersection_directions = self.next_intersection(image_timestamp)
		rospy.loginfo(f"Next intersection : distance = {intersection_distance}, directions = {bin(intersection_directions) if intersection_directions is not None else None}")

		# Close enough to the intersection, switch navigation mode
		if intersection_distance is not None and intersection_distance < self.parameters["intersection"]["mode-switch-distance"]:
			try:
				# The intersection does not allow the chosen direction : go forth, keep for next time
				if not self.next_direction & intersection_directions:
					if intersection_directions & Direction.FORWARD:
						self.switch_intersection(NavigationMode.FORWARD_SKIP, image_timestamp, intersection_distance)
					else:
						# Only left or right, no direction chosen -> wait for input
						self.switch_panic(NavigationMode.PANIC_NO_DIRECTION)
				
				# Switch to the relevant intersection mode
				elif self.next_direction == Direction.LEFT:
					self.switch_intersection(NavigationMode.LEFT_TURN, image_timestamp, intersection_distance)
				elif self.next_direction == Direction.RIGHT:
					self.switch_intersection(NavigationMode.RIGHT_TURN, image_timestamp, intersection_distance)
				elif self.next_direction == Direction.FORWARD:
					self.switch_intersection(NavigationMode.FORWARD_SKIP, image_timestamp, intersection_distance)
			except Exception as exc:
				self.switch_panic(NavigationMode.PANIC_EXCEPTION, exc)
	
	def switch_cruise(self, image_timestamp):
		"""Switch to cruise navigation mode"""
		self.navigation_mode = NavigationMode.CRUISE
		self.next_direction = Direction.FORWARD  # Go forward by default
		self.last_lane_rejoin = image_timestamp
		rospy.loginfo(f"Switching navigation mode : {self.navigation_mode}")
	
	def switch_intersection(self, navigation_mode, image_timestamp, intersection_distance):
		"""Switch to an intersection navigation mode, and compute the necessary trajectories
		   - navigation_mode : NavigationMode : New navigation mode to apply
		"""
		assert navigation_mode in (NavigationMode.FORWARD_SKIP, NavigationMode.LEFT_TURN, NavigationMode.RIGHT_TURN)

		# Switch mode and do all necessary operations according to the direction
		rospy.loginfo(f"Switching navigation mode : {navigation_mode}")
		self.navigation_mode = navigation_mode
		if self.navigation_mode == NavigationMode.FORWARD_SKIP:
			self.build_forward_skip_trajectory(image_timestamp, intersection_distance)
		elif self.navigation_mode == NavigationMode.RIGHT_TURN:
			self.build_right_turn_trajectory(image_timestamp, intersection_distance)
		elif self.navigation_mode == NavigationMode.LEFT_TURN:
			self.build_left_turn_trajectory(image_timestamp, intersection_distance)
		
		# Clear the intersection hints for next time
		self.intersection_hints.clear()

		# Clear the trajectory buffers, the old trajectories won’t be relevant anymore after the intersection
		self.trajectory_buffer.clear()
		self.trajectory_scores.clear()
		self.trajectory_timestamps.clear()

		# Publish the intersection navigation trajectory
		self.publish_trajectory(self.current_trajectory.transpose(), self.current_trajectory_timestamp)
	
	def switch_panic(self, navigation_mode, exc=None):
		"""Switch into some panic mode, in case something goes unrecoverably wrong
		   - navigation_mode : NavigationMode : Panic navigation mode to apply
		   - exc             : Exception      : Exception to display, or None"""
		self.navigation_mode = navigation_mode
		if navigation_mode == NavigationMode.PANIC_NO_DIRECTION:
			rospy.logerr("!!!! PANIC : NO DIRECTION CHOSEN, FORWARD NOT POSSIBLE")
			rospy.logerr("     ------> CHOOSE A DIRECTION TO CONTINUE")
		elif navigation_mode == NavigationMode.PANIC_UNSUPPORTED:
			rospy.logerr("!!!! PANIC : UNSUPPORTED OPERATION")
			rospy.logerr("     ------> SORRY")
		elif navigation_mode == NavigationMode.PANIC_EXCEPTION:
			rospy.logerr("!!!! PANIC : AN EXCEPTION HAPPENED IN A CRITICAL PROCEDURE")
			rospy.logerr("     ------> SORRY")
		elif navigation_mode == NavigationMode.PANIC_CORE_BREACH:
			rospy.logerr("!!!! PANIC : AN UNRECOVERABLE ERROR HAPPENED")
			rospy.logerr("     ------> SORRY")
		if exc is not None:
			rospy.logerr(exc)
		# TODO : STOP THE CAR
	
	def is_panic(self):
		return 500 <= self.navigation_mode.value < 600
	
	#                         ╔══════════════════╗                          #
	# ════════════════════════╣ IMAGE PROCESSING ╠═════════════════════════ #
	#                         ╚══════════════════╝                          #

	def preprocess_image(self, image, target_to_camera):
		"""Preprocess the image receive from the camera
		   - image            : ndarray[y, x, 3] : RGB image received from the camera
		   - target_to_camera : ndarray[4, 4]    : 3D homogeneous transform matrix from the target (road) frame to the camera frame
		<---------------------- ndarray[v, u]    : Full grayscale bird-eye view (mostly for visualisation)
		<---------------------- ndarray[v, u]    : Fully preprocessed bird-eye view (binarized, edge-detected)
		<---------------------- float            : Scale factor, multiply by this to convert lengths from pixel to metric in the target frame
		"""
		# Convert the image to grayscale
		grayimage = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

		# Binarize the image. First a gaussian blur is applied to reduce noise,
		img_blur = cv.GaussianBlur(grayimage, (7, 7), 1.5)
		
		# Project in bird-eye view
		# then a gaussian adaptive thresholding is applied to reduce the influence of lighting changes
		birdeye, scale_factor = fish2bird.to_birdeye(img_blur, self.camera_to_image, target_to_camera, self.distortion_parameters[0], self.birdeye_range_x, self.birdeye_range_y, self.parameters["birdeye"]["birdeye-size"], interpolate=True, flip_y=True)
		be_binary = cv.adaptiveThreshold(birdeye, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, self.parameters["preprocess"]["threshold-window"], self.parameters["preprocess"]["threshold-bias"])

		# The adaptive binarization makes the borders white, mask them out
		mask = cv.erode(np.uint8(birdeye > 0), cv.getStructuringElement(cv.MORPH_RECT, (self.parameters["preprocess"]["threshold-window"]//2 + 2, self.parameters["preprocess"]["threshold-window"]//2 + 2)))
		be_binary *= mask

		# Apply an opening operation to eliminate a few artifacts and better separate blurry markings
		open_kernel_size = self.parameters["preprocess"]["open-kernel-size"]
		open_kernel = cv.getStructuringElement(cv.MORPH_RECT, (open_kernel_size, open_kernel_size))
		be_binary = cv.morphologyEx(be_binary, cv.MORPH_OPEN, open_kernel)

		# Edge detection to get the 1-pixel wide continuous curves required by the following operations
		be_binary = cv.Canny(be_binary, 50, 100)
		return birdeye, be_binary, scale_factor
	
	# ══════════════════════════ LANE DETECTION ═══════════════════════════ #

	def estimate_main_angle(self, lines):
		"""Estimate the main angle of detected lines in some area ahead of the vehicle,
		   to better estimate the current lane’s expected position
		   - lines : list<ndarray[2, N]> : Discrete curves in the image
		<----------- float               : Principal angle of those lines, in radians"""
		# First get the local angles of each discrete curve segment
		angles = []
		for line in lines:
			# Filter out the points that are not in the "local area" of estimation
			local_filter = ((-self.parameters["fuzzy-lines"]["local-area-x"] < line[0]) & (line[0] < self.parameters["fuzzy-lines"]["local-area-x"]) &
			                (0 < line[1]) & (line[1] < self.parameters["fuzzy-lines"]["local-area-y"]))
			# Less than a segment remaining
			if local_filter.sum() < 2:
				continue
			
			line_angles = np.arctan2(np.gradient(line[1, local_filter]), np.gradient(line[0, local_filter]))
			angles.append(line_angles)
		# No data to estimate from : just assume π/2 (straight ahead)
		if len(angles) == 0:
			return np.pi/2
		angles = np.concatenate(angles, axis=0)
		
		# Remove the angles that are a tad too extreme
		angles = angles[(self.parameters["fuzzy-lines"]["main-angle-cut"] < angles) & (angles < np.pi - self.parameters["fuzzy-lines"]["main-angle-cut"])]
		if angles.size == 0:
			return np.pi/2

		# Return the angle of maximal density among the extracted angles
		density_model = KernelDensity(kernel="epanechnikov", bandwidth=0.1)
		density_model.fit(angles.reshape(-1, 1))
		angle_density = density_model.score_samples(angles.reshape(-1, 1))
		return angles[np.argmax(angle_density)]

	def detect_lane(self, be_binary, scale_factor, image_timestamp):
		"""Find the best candidate for the current lane
		   - be_binary       : ndarray[y, x] : Preprocessed bird-eye view
		   - scale_factor    : float         : Scale factor from pixel to metric
		   - image_timestamp : rospy.Time    : Timestamp at which the image was captured
		<--------------------- ndarray[2, L] : Left line as a discrete curve in pixel coordinates, or None
		<--------------------- ndarray[2, R] : Right line as a discrete curve in pixel coordinates, or None (both can be defined or None at the same time)
		<--------------------- float         : Reliability score of the left line in [0, 1], or None
		<--------------------- float         : Reliability score of the right line in [0, 1], or None
		"""
		# Build the current parameters of the filter_lines function
		parameters = self.parameters["find-lines-parameters"].copy()
		parameters["max_curvature"] = parameters["max_curvature_metric"] * scale_factor
		del parameters["max_curvature_metric"]

		# Extract the discrete curves in the image, then the road markings, and finally the potential lane markings
		branches = [branch.astype(float) for branch in linetrack.extract_branches(be_binary)]
		
		# No line detected at all
		if len(branches) == 0:
			rospy.logwarn("No lines found")
			return None, None, None, None
		
		markings, branches = trajectorybuild.find_markings(be_binary.shape, branches, scale_factor, self.parameters["environment"]["crosswalk-width"], self.parameters["markings"]["size-tolerance"])
		for crosswalk in markings["crosswalks"]:
			if len(crosswalk) >= 3:
				target_bands = [self.birdeye_to_target(be_binary, band) for band in crosswalk]
				band_centroids = np.asarray([np.mean(band, axis=1) for band in target_bands])
				crosswalk_centroid = np.concatenate((np.mean(band_centroids, axis=0), [0]))
				self.add_intersection_hint(IntersectionHint("marking", "crosswalk", crosswalk_centroid, image_timestamp, 1))

		lines = linetrack.filter_lines(branches, **parameters)

		# Flip the lines so that they start at the bottom of the image,
		for i, line in enumerate(lines):
			if line[1, 0] < line[1, -1]:
				line = np.flip(line, axis=1)
				lines[i] = line
		
		# Convert all curves from pixel coordinates to metric coordinates in the local road frame 
		target_lines = [self.birdeye_to_target(be_binary, line) for line in lines]

		# Estimate the main angle of those lines
		# If we are on a road, chances are, a lot of curve segments belong to lane markings
		# In the few meters ahead of the vehicle, those have most likely approximately all the same angle
		# Thus, to estimate the angle of the lane relative to the vehicle, to estimate the expected position of the current
		# lane markings better, we just have to take the angle where the angles of curve segments are most concentrated 
		main_angle = self.estimate_main_angle(target_lines)

		# Build the fuzzy logic base variables for each line
		(forward_distance, left_line_distance, right_line_distance, line_lengths, parallel_distance, parallel_angles
			) = trajectorybuild.line_parameters(target_lines, self.parameters["environment"]["lane-width"], main_angle, (self.parameters["birdeye"]["roi-y"] + self.parameters["fuzzy-lines"]["local-area-y"]) / 2, self.parameters["fuzzy-lines"]["vertical-angle-tolerance"])

		# Cruise mode : Detect the current lane, and fall back to a single marking if no sufficiently correct full lane is found
		if self.navigation_mode == NavigationMode.CRUISE:
			left_line_index, right_line_index, left_line_score, right_line_score = self.detect_full_lane(forward_distance, left_line_distance, right_line_distance, line_lengths, parallel_distance, parallel_angles, fallback=True)
		# Intersection modes, AFTER the rejoin distance has been reached :
		# The vehicle must now catch the next lane to follow, so to maximize the chances of getting it right,
		# force it to use only FULL lanes (left and right), or fail and wait for the next image 
		elif self.navigation_mode in (NavigationMode.FORWARD_SKIP, NavigationMode.LEFT_TURN, NavigationMode.RIGHT_TURN):
			left_line_index, right_line_index, left_line_score, right_line_score = self.detect_full_lane(forward_distance, left_line_distance, right_line_distance, line_lengths, parallel_distance, parallel_angles, fallback=False)
		
		left_line = target_lines[left_line_index] if left_line_index is not None else None
		right_line = target_lines[right_line_index] if right_line_index is not None else None

		if self.parameters["node"]["visualize"]:
			self.visualisation.update_line_detection(be_binary, lines, left_line_index, right_line_index, markings)
		return left_line, right_line, left_line_score, right_line_score


	def detect_full_lane(self, forward_distance, left_line_distance, right_line_distance, line_lengths, parallel_distance, parallel_angles, fallback=True):
		"""Detect a full lane (left and right marking) among a list of discrete curves, given some statistics about them
		   - forward_distance    : ndarray[N]    : Distance of the first valid point of the curve to the vehicle. The closer the better
		   - left_line_distance  : ndarray[N]    : Distance of the first valid point to the expected left marking. The closer the better
		   - right_line_distance : ndarray[N]    : Distance of the first valid point to the expected right marking. The closer the better
		   - line_lengths        : ndarray[N]    : Lengths of the curve from the first valid point. The longer the better
		   - parallel_distance   : ndarray[N, N] : Difference of the mean orthogonal distance between both curves to the expected lane width, in terms of lane widths, for each pair of curve. The lower the better
		   - parallel_angles     : ndarray[N, N] : Mean angle between the vectors of both curves, for each pair of curves, in radians. The lower the better
		   - fallback            : bool          : If True, when no sufficiently high-quality lane is found, fall back to finding only one of both markings instead.
		<------------------------- int           : Index of the best left marking found among the initial list of curves, or None
		<------------------------- int           : Index of the best right marking found among the initial list of curves, or None (both can be defined or None at the same time)
		<------------------------- float         : Reliability score of the left line in [0, 1], or None
		<------------------------- float         : Reliability score of the right line in [0, 1], or None
		"""
		# Put the variables in a [5, N, N] array for the fuzzy logic module
		# We need 2D variables, for each pair of curves, so we need to combine their 1D values
		# Forward distance and line length use the usual arithmetic mean, the distance to the expected markings
		# take the best combination of left and right among the pair, and takes the maximum distance among the combination
		FD_columns, FD_rows = np.meshgrid(forward_distance, forward_distance)
		LD_columns, LD_rows = np.meshgrid(left_line_distance, left_line_distance)
		RD_columns, RD_rows = np.meshgrid(right_line_distance, right_line_distance)
		LL_columns, LL_rows = np.meshgrid(line_lengths, line_lengths)
		lane_variables = np.asarray(((FD_columns + FD_rows) / 2,
		                             np.minimum(np.maximum(LD_columns, RD_rows), np.maximum(RD_columns, LD_rows)),
		                             (LL_columns + LL_rows) / 2,
		                             parallel_distance,
		                             parallel_angles))
		
		# Get the best combination and its score with the fuzzy system
		best_y, best_x, best_score = self.lane_system.fuzzy_best(lane_variables)
		rospy.loginfo(f"Best lane score {best_score} for combination {[best_y, best_x]}")
		
		# No good enough lane detected : fall back to single lines if so parameterized
		if best_score < self.parameters["fuzzy-lines"]["lane-selection-threshold"]:
			if fallback:
				rospy.logwarn("No viable lane detected : resorting to single lines")
				return self.detect_any_line(forward_distance, left_line_distance, right_line_distance, line_lengths)
			else:
				rospy.logwarn("No viable lane detected, fail")
				return None, None, None, None
		else:
			# Take the best configuration of left and right line for the combination found,
			# This is made overly complicated because we want to get the best *individual* assignment
			# to each line, not just some least-square solution, because sometimes the full lane score is okay
			# but one line is spot-on while the other is trash
			#
			# For example, when it detects the adjacent lane, the it has found the left marking of the current lane,
			# and the left marking of the other lane, that is completely wrong for the current lane
			#
			# In that case, a least-squares solution would assign the good marking to the right, and the bad to the left,
			# whereas we want to assign the good marking to the left of the current lane, where it belongs, and ditch
			# the bad marking afterwards
			if right_line_distance[best_y] < right_line_distance[best_x]:
				if left_line_distance[best_y] < left_line_distance[best_x]:
					if left_line_distance[best_y] < right_line_distance[best_y]:
						left_line_index, right_line_index = best_y, best_x
					else:
						left_line_index, right_line_index = best_x, best_y
				else:
					left_line_index, right_line_index = best_x, best_y
			else:
				if left_line_distance[best_x] < left_line_distance[best_y]:
					if left_line_distance[best_x] < right_line_distance[best_x]:
						left_line_index, right_line_index = best_x, best_y
					else:
						left_line_index, right_line_index = best_y, best_x
				else:
					left_line_index, right_line_index = best_y, best_x

			# Use the single-marking fuzzy system to estimate the individual reliability score for both curves
			line_variables = np.asarray((
				([forward_distance[left_line_index]],                    [forward_distance[right_line_index]]),
				([left_line_distance[left_line_index]],                  [right_line_distance[right_line_index]]),
				([line_lengths[left_line_index]],                        [line_lengths[right_line_index]]),
				([parallel_distance[left_line_index, right_line_index]], [parallel_distance[left_line_index, right_line_index]]),
				([parallel_angles[left_line_index, right_line_index]],   [parallel_angles[left_line_index, right_line_index]]),
			))
			line_scores = self.lane_system.fuzzy_scores(line_variables)
			left_line_score = line_scores[0, 0]
			right_line_score = line_scores[1, 0]

			rospy.loginfo(f"Respective line scores : {best_score} -> {left_line_score}, {right_line_score}")

			# Sometimes the full lane score is just okayish, because one of the curves is spot-on but the other is blatantly wrong
			# In that case, eliminate the bad one and continue only with the good one
			if left_line_score < self.parameters["fuzzy-lines"]["single-line-selection-threshold"]:
				rospy.logwarn("Found full lane but the left line’s score is too low, ditching the left line")
				return None, right_line_index, None, best_score
			elif right_line_score < self.parameters["fuzzy-lines"]["single-line-selection-threshold"]:
				rospy.logwarn("Found full lane but the right line’s score is too low, ditching the right line")
				return left_line_index, None, best_score, None
			else:
				return left_line_index, right_line_index, left_line_score, right_line_score
	
	def detect_any_line(self, forward_distance, left_line_distance, right_line_distance, line_lengths):
		"""Wrapper to find the best single marking using the minimum of the distances to the left and right expected markings
		   See `.detect_single_line` for details
		"""
		best_line_index, best_score = self.detect_single_line(forward_distance, np.minimum(right_line_distance, left_line_distance), line_lengths)
		if best_line_index is None:
			return None, None, None, None
		elif right_line_distance[best_line_index] < left_line_distance[best_line_index]:
			return None, best_line_index, None, best_score
		else:
			return best_line_index, None, best_score, None
	
	def detect_single_line(self, forward_distance, line_distance, line_lengths):
		"""Detect the best single marking to constrain the current lane
		   - forward_distance : ndarray[N] : Distance of the first valid point of the curve to the vehicle. The closer the better
		   - line_distance    : ndarray[N] : Distance of the first valid point to the expected marking. The closer the better
		   - line_lengths     : ndarray[N] : Lengths of the curve from the first valid point. The longer the better
		<---------------------- int        : Index of the best marking found among the initial list of curves, or None
		<---------------------- float      : Reliability score of the line in [0, 1], or None
		"""
		# Emulate the 2D variables with the best value, such that it has no impact (malus 0)
		single_line_variables = np.asarray((forward_distance.reshape(-1, 1),
											line_distance.reshape(-1, 1),
											line_lengths.reshape(-1, 1),
											np.ones((forward_distance.size, 1)) * self.parameters["fuzzy-lines"]["centers"]["parallel-distances"][0],
											np.ones((forward_distance.size, 1)) * self.parameters["fuzzy-lines"]["centers"]["parallel-angles"][0]))
		best_line_index, best_x, best_score = self.lane_system.fuzzy_best(single_line_variables)

		rospy.loginfo(f"Best single line score {best_score}")
		if best_score < self.parameters["fuzzy-lines"]["single-line-selection-threshold"]:
			rospy.logerr("No viable single line detected")
			return None, None
		else:
			return best_line_index, best_score

	# ═══════════════════ CRUISE TRAJECTORY ESTIMATION ════════════════════ #

	def localize_trajectories(self, lines, scores, timestamps, target_timestamp):
		"""Transform the given trajectories taken in the local frame at the given timestamps, to the local frame at the target timestamp,
		   Then cut and extrapolate such that their first point is at the distance where we want to start the trajectory estimation
		   - lines            : list<ndarray[2, N]> : List of discrete curves in the map frame
		   - scores           : list<ndarray[N]>    : Scores for each point of each curve
		   - timestamps       : list<rospy.Time>    : Timestamp at which each curve was acquired
		   - target_timestamp : rospy.Time          : Timestamp to localize the trajectories into
		<------------------- list<ndarray[2, M]>    : The input curves converted, cut and extended as needed
		<------------------- list<ndarray[M]>       : Scores for each point of each local curve
		<------------------- rospy.Time             : Unbiased target timestamp. This is an artifact of the time when the simulator gave incoherent timestamps,
		                                              on the latest version it is same as target_timestamp"""
		transforms, start_unbiased, target_unbiased = self.get_map_transforms(np.asarray(timestamps), target_timestamp)
		extended_lines = []
		point_scores = []
		for line, line_scores, transform in zip(lines, scores, transforms):
			if line is None or line_scores is None:
				continue

			# Convert to homogeneous 3D coordinates, then transform to the local frame and filter out the points that are behind the start point
			local_line = (transform @ np.vstack((line, np.zeros((1, line.shape[1])), np.ones((1, line.shape[1])))))[:2]
			cut_line = local_line[:, local_line[1] >= self.parameters["trajectory"]["trajectory-start"]]
			if cut_line.shape[1] > 1:
				if cut_line[1, 0] > self.parameters["trajectory"]["trajectory-start"]:
					# Now add a point to the curve in the continuity of its initial vector so that it starts at the start coordinate
					initial_vector = cut_line[:, 1] - cut_line[:, 0]
					initial_distance = (cut_line[1, 0] - self.parameters["trajectory"]["trajectory-start"]) / initial_vector[1]
					initial_point = cut_line[:, 0] - initial_vector * initial_distance
					extended_line = np.concatenate((initial_point.reshape(2, 1), cut_line), axis=1)

					# Compute the score of the new point
					extension_score = line_scores[0] * max(0, 1 - np.exp((initial_distance - self.parameters["trajectory"]["line-reliability-range"]) / (self.parameters["trajectory"]["line-reliability-dampening"] * self.parameters["trajectory"]["line-reliability-extension-penalty"])))
					line_scores = np.concatenate(([extension_score], line_scores))
				else:
					extended_line = cut_line
				extended_lines.append(extended_line)
				point_scores.append(line_scores)
		return extended_lines, point_scores, target_unbiased

	def compile_trajectory(self, timestamp, left_line, left_score, right_line, right_score, viz=None):
		"""Estimate a trajectory from the lane markings extracted from a single frame, and add it to the trajectory history buffer
		   - timestamp     : rospy.Time    : Timestamp of the received image
		   - left_line     : ndarray[2, L] : Left line as a discrete curve, or None
		   - left_score    : float         : Reliability score of the left line, or None
		   - right_line    : ndarray[2, R] : Right line as a discrete curve, or None
		   - right_score   : float         : Reliability score of the right line, or None
		   - viz           : ndarray[y, x] : Visualisation image, can be None
		"""
		# Compute the trajectory estimate according to the left line
		if left_line is not None and left_line.size > 0 and left_line.shape[1] >= 2:
			left_scores = trajectorybuild.init_scores(left_line, left_score, self.parameters["trajectory"]["line-reliability-range"], self.parameters["trajectory"]["line-reliability-dampening"])
			left_estimate, left_estimate_scores = trajeometry.dilate_curve(left_line, self.parameters["environment"]["lane-width"]/2, -1, scores=left_scores)
			left_estimate, left_estimate_scores = trajeometry.strip_angles(left_estimate, np.pi/3, scores=left_estimate_scores)

			# Visualize the line and estimate
			if viz is not None:
				viz_points = self.target_to_birdeye(viz, left_line).astype(int)
				for i in range(viz_points.shape[1]):
					cv.drawMarker(viz, viz_points[:, i], (0, 255, int(left_scores[i]*255)), cv.MARKER_CROSS, 4)  # Red

				viz_points = self.target_to_birdeye(viz, left_estimate).astype(int)
				for i in range(viz_points.shape[1]):
					cv.drawMarker(viz, viz_points[:, i], (12, 255, int(left_estimate_scores[i]*255)), cv.MARKER_CROSS, 4)  # Orange
		else:
			left_estimate = None

		# Compute the trajectory estimate according to the right line
		if right_line is not None and right_line.size > 0 and right_line.shape[1] >= 2:
			right_scores = trajectorybuild.init_scores(right_line, right_score, self.parameters["trajectory"]["line-reliability-range"], self.parameters["trajectory"]["line-reliability-dampening"])
			right_estimate, right_estimate_scores = trajeometry.dilate_curve(right_line, self.parameters["environment"]["lane-width"]/2, 1, scores=right_scores)
			right_estimate, right_estimate_scores = trajeometry.strip_angles(right_estimate, np.pi/3, scores=right_estimate_scores)

			# Visualize the line and estimate
			if viz is not None:
				viz_points = self.target_to_birdeye(viz, right_line).astype(int)
				for i in range(viz_points.shape[1]):
					cv.drawMarker(viz, viz_points[:, i], (90, 255, int(right_scores[i]*255)), cv.MARKER_CROSS, 4)  # Cyan

				viz_points = self.target_to_birdeye(viz, right_estimate).astype(int)
				for i in range(viz_points.shape[1]):
					cv.drawMarker(viz, viz_points[:, i], (125, 255, int(right_estimate_scores[i]*255)), cv.MARKER_CROSS, 4)  # Blue
		else:
			right_estimate = None
		
		# If only one estimate is valid, use it directly, otherwise compile the trajectory from both estimates
		if left_estimate is not None:
			if right_estimate is not None:
				trajectory, trajectory_scores = trajectorybuild.compile_line([left_estimate, right_estimate], [left_estimate_scores, right_estimate_scores], self.parameters["trajectory"]["trajectory-score-threshold"], (0, self.parameters["trajectory"]["trajectory-start"]), self.parameters["trajectory"]["trajectory-step"])
			else:
				trajectory, trajectory_scores = left_estimate, left_estimate_scores
		elif right_estimate is not None:
			trajectory, trajectory_scores = right_estimate, right_estimate_scores
		else:
			trajectory, trajectory_scores = None, None

		# If the trajectory estimate is valid, smooth it and add it to the history buffer
		if trajectory is not None and trajectory.shape[1] > 3:
			filtered_trajectory = trajeometry.savgol_filter(trajectory, trajeometry.savgol_window(7, trajectory.shape[1]), 2)

			self.trajectory_buffer.append(filtered_trajectory)
			self.trajectory_scores.append(trajectory_scores)
			self.trajectory_timestamps.append(timestamp)

			# Discard older trajectories
			while len(self.trajectory_buffer) > self.parameters["trajectory"]["history-size"]:
				self.trajectory_buffer.pop(0)
				self.trajectory_scores.pop(0)
				self.trajectory_timestamps.pop(0)
			
			# Make the transform service discard velocity data older than what will be useful from this point on
			self.drop_velocity(self.trajectory_timestamps[0])

			if viz is not None:
				viz_points = self.target_to_birdeye(viz, filtered_trajectory).transpose().astype(int)
				cv.polylines(viz, [viz_points], False, (30, 255, 255), 2)  # Yellow
	
	# ════════════════ INTERSECTION TRAJECTORY ESTIMATION ═════════════════ #

	def build_forward_skip_trajectory(self, image_timestamp, intersection_distance):
		"""Precompute the trajectory to follow to go forward through an intersection
		   - image_timestamp : rospy.Time : Timestamp at which to estimate the trajectory
		   - intersection_distance : float : Distance remaining until the intersection (most likely negative, the vehicle is already on the intersection)
		"""
		self.rejoin_distance = self.parameters["intersection"]["default-rejoin-distance"]
		transforms, start_unbiased, target_unbiased = self.get_map_transforms(np.asarray(self.trajectory_timestamps), image_timestamp)
		local_lines = [(transform @ np.vstack((trajectory, np.zeros((1, trajectory.shape[1])), np.ones((1, trajectory.shape[1])))))[:2] for transform, trajectory in zip(transforms, self.trajectory_buffer)]
		cut_lines = [local_line[:, local_line[1] < intersection_distance] for local_line in local_lines]
		main_angle = self.estimate_main_angle(cut_lines)

		# Just output a straight trajectory for as long as necessary based on that main angle		
		max_length = int(100 / self.parameters["trajectory"]["trajectory-step"])
		self.current_trajectory = np.asarray((np.arange(0, max_length, 1) * self.parameters["trajectory"]["trajectory-step"] * np.cos(main_angle),
											  np.arange(0, max_length, 1) * self.parameters["trajectory"]["trajectory-step"] * np.sin(main_angle)))
		# self.current_trajectory = np.asarray((np.zeros(max_length), np.arange(0, max_length, 1) * self.parameters["trajectory"]["trajectory-step"]))
		self.current_trajectory_timestamp = image_timestamp
	
	def build_right_turn_trajectory(self, image_timestamp, intersection_distance):
		"""Precompute the trajectory to follow to turn right at an intersection
		   - image_timestamp : rospy.Time : Timestamp at which to estimate the trajectory
		   - intersection_distance : float : Distance remaining until the intersection (most likely negative, the vehicle is already on the intersection)
		"""
		self.rejoin_distance = self.parameters["intersection"]["default-rejoin-distance"]

		# Get the previously estimated trajectories, transform them into the local frame,
		# and keep only the parts that are within the intersection
		# Then get their mean curvature, that is used afterwards to estimate the turn curvature
		transforms, start_unbiased, target_unbiased = self.get_map_transforms(np.asarray(self.trajectory_timestamps), image_timestamp)
		curvatures = []
		for i, (transform, trajectory) in enumerate(zip(transforms, self.trajectory_buffer)):
			local_line = (transform @ np.vstack((trajectory, np.zeros((1, trajectory.shape[1])), np.ones((1, trajectory.shape[1])))))[:2]
			cut_line = local_line[:, local_line[1] > intersection_distance]
			if cut_line.shape[1] >= 3:
				curvatures.append(trajeometry.mean_curvature(cut_line))
		curvatures = np.asarray(curvatures)
		
		# Now estimate the curvature of the turn by taking the value of maximal density among those curvatures
		density_model = KernelDensity(kernel="epanechnikov", bandwidth=0.05)
		density_model.fit(np.asarray(curvatures).reshape(-1, 1))
		curvature_density = density_model.score_samples(curvatures.reshape(-1, 1))
		marking_curvature = curvatures[np.argmax(curvature_density)]

		# The curvature is in rad/m, so just take its inverse to get the curvature radius
		trajectory_radius = 1 / marking_curvature

		# If it’s obviously wrong, clip it to default values
		# This won’t do much good, but well, let’s see where it goes and if it’s really awful we’ll make it panic
		if trajectory_radius < self.parameters["intersection"]["min-turn-radius"]:
			rospy.logwarn(f"Absurdly small curve radius found ({trajectory_radius:.3f}m), clipping to {self.parameters['intersection']['min-turn-radius']}")
			trajectory_radius = self.parameters["intersection"]["min-turn-radius"]
		if trajectory_radius > self.parameters["intersection"]["max-turn-radius"]:
			rospy.logwarn(f"Absurdly large curve radius found ({trajectory_radius:.3f}m), clipping to {self.parameters['intersection']['max-turn-radius']}")
			trajectory_radius = self.parameters["intersection"]["max-turn-radius"]

		# And from that, compute the trajectory as a quarter circle to the right with that radius
		angle_step = self.parameters["trajectory"]["trajectory-step"] / trajectory_radius
		angles = np.flip(np.arange(np.pi/2, np.pi, angle_step))
		self.current_trajectory = np.asarray((trajectory_radius*(1 + np.cos(angles)), trajectory_radius*np.sin(angles) + intersection_distance))
		self.current_trajectory_timestamp = image_timestamp
	
	def build_left_turn_trajectory(self, image_timestamp, intersection_distance):
		"""Precompute the trajectory to follow to turn right at an intersection
		   - image_timestamp : rospy.Time : Timestamp at which to estimate the trajectory
		   - intersection_distance : float : Distance remaining until the intersection (most likely negative, the vehicle is already on the intersection)
		"""
		# Basically, we’re doing pretty much the same thing as a right turn
		# except the final trajectory is one radius further to accomodate the additional lane to pass,
		# and of course goes to the left
		self.rejoin_distance = self.parameters["intersection"]["default-rejoin-distance"]

		transforms, start_unbiased, target_unbiased = self.get_map_transforms(np.asarray(self.trajectory_timestamps), image_timestamp)
		curvatures = []
		for i, (transform, trajectory) in enumerate(zip(transforms, self.trajectory_buffer)):
			local_line = (transform @ np.vstack((trajectory, np.zeros((1, trajectory.shape[1])), np.ones((1, trajectory.shape[1])))))[:2]
			cut_line = local_line[:, local_line[1] > intersection_distance]
			if cut_line.shape[1] >= 3:
				curvatures.append(trajeometry.mean_curvature(cut_line))
		curvatures = np.asarray(curvatures)
		
		# Trajectory radius estimation from density among the curvature values
		density_model = KernelDensity(kernel="epanechnikov", bandwidth=0.05)
		density_model.fit(curvatures.reshape(-1, 1))
		curvature_density = density_model.score_samples(curvatures.reshape(-1, 1))
		marking_curvature = curvatures[np.argmax(curvature_density)]

		trajectory_radius = 1 / marking_curvature
		if trajectory_radius < self.parameters["intersection"]["min-turn-radius"]:
			rospy.logwarn(f"Absurdly small curve radius found ({trajectory_radius:.3f}m), clipping to {self.parameters['intersection']['min-turn-radius']}")
			trajectory_radius = self.parameters["intersection"]["min-turn-radius"]
		if trajectory_radius > self.parameters["intersection"]["max-turn-radius"]:
			rospy.logwarn(f"Absurdly large curve radius found ({trajectory_radius:.3f}m), clipping to {self.parameters['intersection']['max-turn-radius']}")
			trajectory_radius = self.parameters["intersection"]["max-turn-radius"]

		angle_step = self.parameters["trajectory"]["trajectory-step"] / trajectory_radius
		angles = np.arange(0, np.pi/2, angle_step)
		self.current_trajectory = np.asarray((trajectory_radius*(np.cos(angles) - 1), trajectory_radius*np.sin(angles) + self.parameters["environment"]["lane-width"] + intersection_distance))
		self.current_trajectory_timestamp = image_timestamp
	
	# ═══════════════════ FINAL TRAJECTORY CONSTRUCTION ═══════════════════ #

	def update_trajectory(self, target_timestamp, viz=None):
		"""Update the current trajectory to the given timestamp, using the trajectory history buffers
		   - target_timestamp : rospy.Time : Timestamp for which the final trajectory must be estimated
		"""
		# Transform the trajectory buffer to the local frame
		local_trajectories, local_scores, target_unbiased = self.localize_trajectories(self.trajectory_buffer, self.trajectory_scores, self.trajectory_timestamps, target_timestamp)

		# Visualization of the per-frame trajectories
		if viz is not None:
			for line, line_scores in zip(local_trajectories, local_scores):
				viz_points = self.target_to_birdeye(viz, line).astype(int)
				for i in range(line.shape[1]):
					cv.drawMarker(viz, viz_points[:, i], (150, 255, int(line_scores[i]*255)), cv.MARKER_CROSS, 4)  # Purple
		
		# Compile a global trajectory from the per-frame ones
		compiled_trajectory, compiled_scores = trajectorybuild.compile_line(local_trajectories, local_scores, self.parameters["trajectory"]["trajectory-score-threshold"], (0, self.parameters["trajectory"]["trajectory-start"]), self.parameters["trajectory"]["trajectory-step"])  # Blue

		# Smooth it and update the trajectory to be published
		if compiled_trajectory is not None and compiled_trajectory.size > 0 and compiled_trajectory.shape[1] > 3:
			compiled_trajectory, compiled_scores = trajeometry.strip_angles(compiled_trajectory, np.pi/2, compiled_scores)
			self.current_trajectory = trajeometry.savgol_filter(compiled_trajectory, trajeometry.savgol_window(7, compiled_trajectory.shape[1]), 2)
			self.current_trajectory_timestamp = target_unbiased

			if viz is not None:
				viz_points = self.target_to_birdeye(viz, self.current_trajectory).transpose().astype(int)
				cv.polylines(viz, [viz_points], False, (60, 255, 255), 2)  # Green
		# Failure : Set it to None such that nothing is published
		else:
			self.current_trajectory = None
			self.current_trajectory_timestamp = None
	
	#                           ╔═══════════════╗                           #
	# ══════════════════════════╣ VISUALISATION ╠══════════════════════════ #
	#                           ╚═══════════════╝                           #

	def viz_intersection_mode(self, viz, scale_factor, image_timestamp, remaining_distance):
		"""Visualization in intersection navigation mode
		   - viz                : ndarray[y, x] : Bird-eye view visualization image
		   - scale_factor       : float         : Scale factor from pixel to metric lengths
		   - image_timestamp    : rospy.Time    : Timestamp to visualize at
		   - remaining_distance : float         : Distance until reaching the rejoin distance
		"""
		transform = self.get_map_transforms([self.current_trajectory_timestamp], image_timestamp)[0][0]
		local_trajectory = (transform @ np.vstack((self.current_trajectory, np.zeros((1, self.current_trajectory.shape[1])), np.ones((1, self.current_trajectory.shape[1])))))[:2]
		viz_trajectory = self.target_to_birdeye(viz, local_trajectory)
		cv.polylines(viz, [viz_trajectory.transpose().astype(int)], False, (60, 255, 255), 2)
		if remaining_distance is not None:
			cv.circle(viz, (viz.shape[1]//2, viz.shape[1]), int(remaining_distance / scale_factor), (0, 255, 255), 1)
	
	#                      ╔═════════════════════════╗                      #
	# ═════════════════════╣ BIRD-EYE VIEW UTILITIES ╠═════════════════════ #
	#                      ╚═════════════════════════╝                      #

	# Those are just wrappers to the fish2bird functions, to use the global parameters
	def target_to_birdeye(self, be_binary, target_points):
		return fish2bird.target_to_output(target_points, self.birdeye_range_x, self.birdeye_range_y, be_binary.shape[0], flip_y=True)[0]

	def birdeye_to_target(self, be_binary, image_points):
		return fish2bird.birdeye_to_target(image_points, self.birdeye_range_x, self.birdeye_range_y, be_binary.shape, flip_y=True)[:2]

	#                    ╔════════════════════════════╗                     #
	# ═══════════════════╣ GLOBAL ACTIVITY PROCEDURES ╠════════════════════ #
	#                    ╚════════════════════════════╝                     #

	def publish_trajectory(self, trajectory_points, trajectory_timestamp):
		"""Publish a trajectory on the output topic
		   - trajectory_points    : ndarray[N, 2] : Points of the trajectory as LINE VECTORS, contrary to the rest of the node
		   - trajectory_timestamp : rospy.Time    : Timestamp at which the trajectory is valid
		"""
		# Add the current position (0, 0) to the trajectory, otherwise the first point might be too far away
		# and the pure pursuit will miss it
		trajectory_points = np.concatenate((np.asarray([[0, 0]]), trajectory_points), axis=0)

		trajectory_array = Float64MultiArray()
		trajectory_array.data = trajectory_points.flatten()
		trajectory_array.layout.data_offset =  0
		dim = []
		dim.append(MultiArrayDimension("points", trajectory_points.shape[0], trajectory_points.shape[0]*trajectory_points.shape[1]))
		dim.append(MultiArrayDimension("coords", trajectory_points.shape[1], trajectory_points.shape[1]))
		trajectory_array.layout.dim = dim

		message = Trajectory()
		message.header = Header(seq=self.trajectory_seq, stamp=trajectory_timestamp, frame_id=self.parameters["node"]["road-frame"])
		message.trajectory = trajectory_array
		self.trajectory_seq += 1

		rospy.loginfo("Publishing a new trajectory")
		self.trajectory_publisher.publish(message)

	def compute_trajectory(self, image, timestamp, image_frame):
		"""Global procedure called each time an image is received
		   Take the image received from the camera, estimate the trajectory from it and publish it if necessary
		   - image       : ndarray[y, x, 3] : RGB image received from the camera
		   - timestamp   : rospy.Time       : Timestamp at which the image has been taken. All operations will attune to this timestamp
		   - image_frame : str              : Name of the TF frame of the camera that has taken the picture"""
		starttime = time.time()

		# Get the transform from the camera to the local vehicle frame (base_link)
		target_to_camera = self.get_transform(self.parameters["node"]["road-frame"], image_frame)
		if target_to_camera is None:
			rospy.logerr("No camera -> road transform found")
			return

		# Preprocess the image
		birdeye, be_binary, scale_factor = self.preprocess_image(image, target_to_camera)

		# The following is made overly complicated because of the visualization that must still be updated even though nothing else must be done
		trajectory_viz = cv.cvtColor(cv.merge((birdeye, birdeye, birdeye)), cv.COLOR_BGR2HSV) if self.parameters["node"]["visualize"] else None
		
		# Intersection mode : Check whether the vehicle has reached the rejoin distance, and if so, try to catch the new lane
		# otherwise, just wait further
		if self.navigation_mode in (NavigationMode.FORWARD_SKIP, NavigationMode.LEFT_TURN, NavigationMode.RIGHT_TURN):
			remaining_distance = self.distance_till_rejoin(timestamp)
			must_detect_trajectory = remaining_distance <= 0
			if remaining_distance > 0:
				rospy.loginfo(f"Waiting for rejoin, {remaining_distance} meters remaining…")
				self.viz_intersection_mode(trajectory_viz, scale_factor, timestamp, remaining_distance)
		else:
			must_detect_trajectory = True

		# In cruise mode or intersect mode when the vehicle needs to catch the new lane, go further
		if must_detect_trajectory:
			left_line, right_line, left_line_score, right_line_score = self.detect_lane(be_binary, scale_factor, timestamp)

			# In intersection mode, if a new full lane has been found, catch it and go back to cruise 
			if self.navigation_mode in (NavigationMode.FORWARD_SKIP, NavigationMode.LEFT_TURN, NavigationMode.RIGHT_TURN):
				if left_line is None or right_line is None:
					rospy.loginfo("No full lane found to rejoin, waiting further…")
					self.viz_intersection_mode(trajectory_viz, scale_factor, timestamp, None)
				else:
					rospy.loginfo("Rejoin lane found, switching back to cruise")
					self.switch_cruise(timestamp)
			
			# In cruise mode, update the trajectory, intersection status, and publish if there is something to publish
			# Do NOT refactor this into an `else`, it must also be done when the vehicle just got out of intersection mode
			if self.navigation_mode == NavigationMode.CRUISE:
				self.compile_trajectory(timestamp, left_line, left_line_score, right_line, right_line_score, trajectory_viz)

				# For turns, it’s beneficial to have the most recent trajectory available,
				# so when in cruise mode, update the intersection status only after having compiled the latest trajectory
				self.update_intersection(timestamp)
						
				if len(self.trajectory_buffer) > 0:
					self.update_trajectory(timestamp, trajectory_viz)

				if self.current_trajectory is not None:
					self.publish_trajectory(self.current_trajectory.transpose(), self.current_trajectory_timestamp)
		
		if trajectory_viz is not None:
			self.visualisation.update_trajectory_construction(cv.cvtColor(trajectory_viz, cv.COLOR_HSV2RGB))

		endtime = time.time()
		self.time_buffer.append(endtime - starttime)
		rospy.loginfo(f"Image handled in {endtime - starttime :.3f} seconds (mean {np.mean(self.time_buffer):.3f}) seconds")

#                          ╔═════════════════════╗                          #
# ═════════════════════════╣ NODE INITIALISATION ╠═════════════════════════ #
#                          ╚═════════════════════╝                          #

if __name__ == "__main__":
	# I’m fed up with scientific notation
	np.set_printoptions(threshold=sys.maxsize, suppress=True)

	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <parameter-file>")
	else:
		# Load the parameters and map
		with open(sys.argv[1], "r") as parameterfile:
			parameters = yaml.load(parameterfile, yaml.Loader)

		# Initialize and start the node
		rospy.init_node(parameters["node"]["trajectory-node-name"])
		node = TrajectoryExtractorNode(parameters)
		rospy.spin()
