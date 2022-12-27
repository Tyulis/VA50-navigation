import json
import numpy as np
from scipy.spatial import cKDTree

import trajeometry

class RoadNetwork (object):
	"""Backend to some basic cartography that allows to retrieve approximately where is the next intersection
	   and in which directions it branches from some position on the map"""
	
	# Directions
	DEAD_END = 0b0000
	FORWARD = 0b0001
	LEFT = 0b0010
	RIGHT = 0b0100
	
	def __init__(self, filename):
		"""Initialize the network from the provided JSON file"""
		with open(filename, "r") as f:
			json_data = json.load(f)
		
		# Make it a little bit more practical for us
		self.segments = {item["guid"]: {"guid": item["guid"], "previous": item["previous"], "next": item["next"],
		                                "adjacent": ([item["inner"]] if "inner" in item else []) + ([item["outer"]] if "outer" in item else []),
		                                "basepoints": np.asarray([(segment["px"], segment["py"]) for segment in item["geometry"]]).transpose(),
										"vectors": np.asarray([(segment["tx"], segment["ty"]) for segment in item["geometry"]]).transpose(),
										"widths": np.asarray([segment["width"] for segment in item["geometry"]])} for item in json_data}
		
		# To squeeze a bit more performance out of this, we precompute the directions in which one can go
		# when going through each side of each segment, and make a lookup of road segments based on the base points of their sub-segments
		# such that we can efficiently search through them with a k-d tree
		basepoints = []
		vectors = []
		self.basepoint_mapping = []
		self.possible_directions = {}

		for guid, item in self.segments.items():
			prev_directions = self._extract_directions(item, [self.segments[id] for id in item["previous"]], side=0)
			next_directions = self._extract_directions(item, [self.segments[id] for id in item["next"]], side=1)
			self.possible_directions[guid] = [prev_directions, next_directions]
			basepoints.append(item["basepoints"])
			vectors.append(item["vectors"])
			self.basepoint_mapping.extend([guid] * item["basepoints"].shape[1])
		
		# For the moment, we don’t consider differences in available directions for adjacent lanes
		# (like the left lane goes to the left and the right lane forward and to the right)
		# So we merge the directions of those
		for guid, item in self.segments.items():
			for adjacent_guid in item["adjacent"]:
				if trajeometry.vector_angle(self.segments[adjacent_guid]["vectors"][:, 0], item["vectors"][:, 0]) < np.pi/6:
					self.possible_directions[guid][0] |= self.possible_directions[adjacent_guid][0]
					self.possible_directions[guid][1] |= self.possible_directions[adjacent_guid][1]
		
		# Now build the base points arrays and the k-d tree lookup
		self.basepoints = np.concatenate(basepoints, axis=1)
		self.vectors = np.concatenate(vectors, axis=1)
		self.basepoint_tree = cKDTree(self.basepoints.transpose())
	
	def best_basepoint(self, point, vector):
		"""Find the closest base point index based on the current position and the current direction
		   (to filter out the lanes that go in the opposite direction, as the measurements aren’t very accurate)
		   - point  : ndarray[2] : Current position on the map
		   - vector : ndarray[2] : Vector that gives the current direction on the map
		<---------- int        : Index of the best base point in `self.basepoints`"""
		# We take the best corresponding angle between the nearest base points
		# In case of tie, as the k-d tree query is sorted by increasing distance, argmin will get the closest one
		distances, indices = self.basepoint_tree.query(point, k=4)
		check_vectors = self.vectors[:, indices]

		# Relative angle of each neighbor’s vector with the current vehicle’s direction
		angles = np.arccos((check_vectors[0]*vector[0] + check_vectors[1]*vector[1]) / (np.linalg.norm(vector) * np.linalg.norm(check_vectors, axis=0)))
		best_angle = np.argmin(angles)
		return indices[best_angle]
	
	def nearby_intersection(self, point, vector):
		"""Get the most likely nearest intersection based on the current position and direction of the vehicle
		   - point  : ndarray[2] : Current position on the map
		   - vector : ndarray[2] : Vector that gives the current direction on the map
		<------------ int        : Direction flags for the next intersection
		<------------ float      : Distance from the advertised vehicle position to the next intersection"""
		best_index = self.best_basepoint(point, vector)
		next_guid = self.basepoint_mapping[best_index]
		opposite = trajeometry.vector_angle(vector, self.vectors[:, best_index]) > np.pi/2
		directions = RoadNetwork.FORWARD
		distance = 0
		# While the current segment doesn’t lead to an intersection, go to the next one and add its length to the distance
		while directions == RoadNetwork.FORWARD:
			segment = self.segments[next_guid]
			directions = self.possible_directions[next_guid]
			if opposite:
				directions = self.possible_directions[next_guid][0]
				distance += np.linalg.norm(point - segment["basepoints"][:, 0])
				if directions == RoadNetwork.FORWARD and len(segment["previous"]) > 0:
					next_guid = segment["previous"][0]
			else:
				directions = self.possible_directions[next_guid][1]
				distance += np.linalg.norm(point - segment["basepoints"][:, -1])
				if directions == RoadNetwork.FORWARD and len(segment["next"]) > 0:
					next_guid = segment["next"][0]
		return directions, distance
		
	def _extract_directions(self, item, following, side):
		"""Find the directions that are available at one end of a road segment
		   - item      : dict<str, ...> : Road segment data
		   - following : list<str>      : GUID of the segments that follow that item in that direction
		   - side      : int            : 0 for previous, 1 for next
		<--------------- int            : Bitflags with the available directions"""
		# When zero, one or >=3 segments follow, the result is obvious
		if len(following) == 0:
			return RoadNetwork.DEAD_END
		if len(following) == 1:
			return RoadNetwork.FORWARD
		elif len(following) >= 3:
			return RoadNetwork.LEFT | RoadNetwork.FORWARD | RoadNetwork.RIGHT
		
		# Two-segments cases are more problematic
		item_basepoints = item["basepoints"]
		item_vectors = item["vectors"]
		if side == 0:
			item_basepoints = np.flip(item_basepoints, axis=1)
			item_vectors = -np.flip(item_vectors, axis=1)
		
		following_angles = []
		for branch in following:
			branch_basepoints = item["basepoints"]
			branch_vectors = item["vectors"]
			if np.linalg.norm(branch_basepoints[:, -1] - item_basepoints[:, -1]) < np.linalg.norm(branch_basepoints[:, 0] - item_basepoints[:, -1]):
				branch_basepoints = np.flip(branch_basepoints, axis=1)
				branch_vectors = -np.flip(branch_vectors, axis=1)
			if trajeometry.vector_angle(item_vectors[:, -1], branch_vectors[:, 0]) > np.pi/2:
				branch_vectors = -branch_vectors
			
			last_angle = trajeometry.vector_angle(item_vectors[:, -1], branch_vectors[:, branch_vectors.shape[1]//2])
			if item_vectors[0, -1]*branch_vectors[1, branch_vectors.shape[1]//2] - item_vectors[1, -1]*branch_vectors[0, branch_vectors.shape[1]//2] < 0:
				last_angle = -last_angle
			following_angles.append(last_angle)
		following_angles.sort()
		if following_angles[0] < 0 and following_angles[1] < 0:
			return RoadNetwork.LEFT | RoadNetwork.FORWARD
		elif following_angles[0] > 0 and following_angles[1] > 0:
			return RoadNetwork.FORWARD | RoadNetwork.RIGHT
		elif -np.pi/6 < following_angles[0] <= 0:
			return RoadNetwork.FORWARD | RoadNetwork.RIGHT
		elif 0 < following_angles[1] <= np.pi/6:
			return RoadNetwork.LEFT | RoadNetwork.FORWARD
		else:
			return RoadNetwork.LEFT | RoadNetwork.RIGHT