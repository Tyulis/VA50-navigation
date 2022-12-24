# distutils: language=c++

"""
Module that does curve extraction from binary images
"""

import numpy as np
from scipy.spatial import cKDTree

import trajeometry

cimport cython
cimport trajeometry
from cython.operator cimport dereference as deref, postincrement as postinc
from libc.math cimport sqrt, acos, exp, M_PI, INFINITY
from libc.stdlib cimport malloc, free
from libcpp.deque cimport deque
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.utility cimport pair
from numpy cimport uint8_t


cdef extern from "<algorithm>" namespace "std":
	T max[T](T a, T b)
	T min[T](T a, T b)


cdef struct _MergeState:
	bint merge    # True => the lines can be merged, False => they don’t
	bint flip1    # Whether to flip the first line before concatenation
	bint flip2    # Whether to flip the second line before concatenation
	double score  # Merge score of the combination

cdef double _acos_clip(double val):
	"""Utility function, when we try to get the angle between two vectors using the dot product,
	   sometimes floating-points shenanigans make the cos value get a little bit outside [-1, 1],
	   so this puts it back into the right range and computes the arccos"""
	if val < -1:
		return acos(-1)
	elif val > 1:
		return acos(1)
	else:
		return acos(val)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _MergeState merge_lines(double[:, :] line1, double[:, :] line2, double merge_max_distance, double angle_diff_threshold, double merge_score_threshold):
	"""Check whether both lines can be merged
	   - line1                    : double[2, M] : First point sequence
	   - line2                    : double[2, N] : Second point sequence
	   - merge_max_distance       : double       : Max distance at which two lines can be merged
	   - angle_diff_threshold     : double       : Maximum difference between the angles of both extremity vectors to be able to merge both lines
	   - merge_score_threshold    : double       : Maximum score required to merge both lines
	<------------------------------ _MergeState  : Resulting merge prediction
	"""

	cdef _MergeState result = _MergeState(merge=False, flip1=False, flip2=False, score=INFINITY)
	cdef Py_ssize_t line1_length = line1.shape[1], line2_length = line2.shape[1]
	
	# The lines could be in any direction
	# We need the points and vectors that directly face each other
	# Those are the squared distance to save a few sqrt() calls
	cdef double extreme_distance_00 = (line1[0, 0] - line2[0, 0])**2 + (line1[1, 0] - line2[1, 0])**2
	cdef double extreme_distance_10 = (line1[0, line1_length-1] - line2[0, 0])**2 + (line1[1, line1_length-1] - line2[1, 0])**2
	cdef double extreme_distance_01 = (line1[0, 0] - line2[0, line2_length-1])**2 + (line1[1, 0] - line2[1, line2_length-1])**2
	cdef double extreme_distance_11 = (line1[0, line1_length-1] - line2[0, line2_length-1])**2 + (line1[1, line1_length-1] - line2[1, line2_length-1])**2
	cdef double distance

	# To concatenate them at the end, the lines must be like 0 --line1--> -1 |----| 0 --line2--> -1
	# So the closest points must be line1[-1] and line2[0]
	# So flip 1 if the closest point is 0, flip 2 if the closest point is -1
	if extreme_distance_00 <= extreme_distance_01 and extreme_distance_00 <= extreme_distance_10 and extreme_distance_00 <= extreme_distance_11:
		distance = sqrt(extreme_distance_00)
		result.flip1 = True
		result.flip2 = False
	elif extreme_distance_01 <= extreme_distance_00 and extreme_distance_01 <= extreme_distance_10 and extreme_distance_01 <= extreme_distance_11:
		distance = sqrt(extreme_distance_01)
		result.flip1 = True
		result.flip2 = True
	elif extreme_distance_10 <= extreme_distance_00 and extreme_distance_10 <= extreme_distance_01 and extreme_distance_10 <= extreme_distance_11:
		distance = sqrt(extreme_distance_10)
		result.flip1 = False
		result.flip2 = False
	else:
		distance = sqrt(extreme_distance_11)
		result.flip1 = False
		result.flip2 = True

	if distance > merge_max_distance:
		return result

	# Extract the closest points and vectors of each line to the other
	# Same flipping logic
	cdef double point1_x, point1_y, vector1_x, vector1_y
	if result.flip1:
		point1_x = line1[0, 0]
		point1_y = line1[1, 0]
		vector1_x = point1_x - line1[0, 1]
		vector1_y = point1_y - line1[1, 1]
	else:
		point1_x = line1[0, line1_length-1]
		point1_y = line1[1, line1_length-1]
		vector1_x = point1_x - line1[0, line1_length-2]
		vector1_y = point1_y - line1[1, line1_length-2]

	cdef double point2_x, point2_y, vector2_x, vector2_y
	if result.flip2:
		point2_x = line2[0, line2_length-1]
		point2_y = line2[1, line2_length-1]
		vector2_x = point2_x - line2[0, line2_length-2]
		vector2_y = point2_y - line2[1, line2_length-2]
	else:
		point2_x = line2[0, 0]
		point2_y = line2[1, 0]
		vector2_x = point2_x - line2[0, 1]
		vector2_y = point2_y - line2[1, 1]
	
	cdef double link_vector_x, link_vector_y
	link_vector_x = point2_x - point1_x
	link_vector_y = point2_y - point1_y

	cdef double vector1_norm = sqrt(vector1_x*vector1_x + vector1_y*vector1_y)
	cdef double vector2_norm = sqrt(vector2_x*vector2_x + vector2_y*vector2_y)
	cdef double link_vector_norm = sqrt(link_vector_x*link_vector_x + link_vector_y*link_vector_y)

	# Compute the angle between both extreme vectors
	# They are pointed toward each other, so one of them is reversed to compare the angles correctly
	# We use the dot product (X₁·X₂ = x₁x₂ + y₁y₂ = ||X₁|| × ||X₂|| × cos(X₁^X₂)) to avoid arctan2 output range issues
	cdef double endpoints_angle = _acos_clip((-vector1_x*vector2_x - vector1_y*vector2_y) / (vector1_norm * vector2_norm))
	cdef double necessary_angle1 = _acos_clip((link_vector_x*vector1_x + link_vector_y*vector1_y) / (vector1_norm * link_vector_norm))
	cdef double necessary_angle2 = _acos_clip((-link_vector_x*vector2_x - link_vector_y*vector2_y) / (vector2_norm * link_vector_norm))

	# Angles not close enough -> ditch this combination	
	if endpoints_angle > angle_diff_threshold:
		return result
	
	cdef double score = distance * max(necessary_angle1, necessary_angle2)
	if score < merge_score_threshold:  # Those are a merge candidate
		result.score = score
		result.merge = True
		return result
	else:
		return result

@cython.wraparound(False)
@cython.boundscheck(False)
cdef int _neighbor_count(uint8_t[:, ::1] remainder, Py_ssize_t base_y, Py_ssize_t base_x) noexcept nogil:
	"""Count the 8-neighbors of a point in `remainder`
	   - remainder      : uint8_t[y, x] : Binary image in which to count the neighbors
	   - base_y, base_x : Py_ssize_t    : Point to count the neighbors of
	<-------------------- int           : Number of 8-neighbors of that point
	"""
	cdef Py_ssize_t count = 0, y, x
	for y in range(base_y - 1, base_y + 2):
		if y < 0 or y >= remainder.shape[0]:
			continue

		for x in range(base_x - 1, base_x + 2):
			if (x == base_x and y == base_y) or x < 0 or x >= remainder.shape[1]:
				continue
			if remainder[y, x] > 0:
				count += 1
	return count

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _extract_branch(uint8_t[:, ::1] remainder, deque[(Py_ssize_t, Py_ssize_t)]& branch, Py_ssize_t base_y, Py_ssize_t base_x, Py_ssize_t init_y, Py_ssize_t init_x):
	"""Extract a single curve in the image, starting from point (base_y, base_x)
	   - remainder : uint8_t[y, x] : Binary values of the image, pixels at 0 are ignored
	   - branch    : deque<(Py_ssize_t, Py_ssize_t)> : Branch to fill with discrete curve points, in order
	   - base_y    : Py_ssize_t : Position at which to start the extraction
	   - base_x    : Py_ssize_t /
	   - init_y    : Py_ssize_t : Position to consider as the previous point on the curve. Use only internal, set to -1 for unused.
	   - init_x    : Py_ssize_t /
	"""
	cdef Py_ssize_t[8][2] neighbors
	cdef Py_ssize_t neighbor_index, y, x, i, best_neighbor1, best_neighbor2
	cdef (Py_ssize_t, Py_ssize_t) last_point
	cdef double sqdistance, max_sqdistance
	cdef deque[(Py_ssize_t, Py_ssize_t)] subbranch
	cdef deque[(Py_ssize_t, Py_ssize_t)].iterator it

	# Return when there are no neighbors remaining
	while True:
		# Add the current point on the branch and remove it from the remainder
		branch.push_back((base_y, base_x))
		remainder[base_y, base_x] = 0

		# Find the 8-neighborhood of the current base point
		neighbor_index = 0
		for y in range(base_y - 1, base_y + 2):
			if y < 0 or y >= remainder.shape[0]:
				continue

			for x in range(base_x - 1, base_x + 2):
				if (x == base_x and y == base_y) or x < 0 or x >= remainder.shape[1]:
					continue
				if remainder[y, x] > 0:
					neighbors[neighbor_index] = [y, x]
					neighbor_index += 1
		
		# No neighbors anymore : quit
		if neighbor_index == 0:
			return
		# One neighbor : just add it and go forth with it
		elif neighbor_index == 1:
			base_y = neighbors[0][0]
			base_x = neighbors[0][1]
		# Two or more neighbors : some selection ought to take place
		elif neighbor_index >= 2:
			# First point on the branch and no init position, this is the initial point on the branch and goes multiple ways
			# Select the two farthest apart neighbors, and add them at both ends of the branch
			if branch.size() == 1 and init_y < 0 and init_y < 0:
				# Select the neighbors that are farthest apart
				max_sqdistance = 0
				best_neighbor1 = best_neighbor2 = -1
				for i in range(neighbor_index):
					for j in range(neighbor_index):
						if i == j:
							continue
						sqdistance = (neighbors[i][0] - neighbors[j][0])**2 + (neighbors[i][1] - neighbors[j][1])**2
						if sqdistance > max_sqdistance:
							max_sqdistance = sqdistance
							best_neighbor1 = i
							best_neighbor2 = j
				# Remove the other neighbors, otherwise they could parasite the following loops
				# Those are just line overthickness, if it is actually a branching curve, it will just
				# clear the first pixel and be caught back later
				for i in range(neighbor_index):
					if i != best_neighbor1 and i != best_neighbor2:
						remainder[neighbors[i][0], neighbors[i][1]] = 0
				# Max square distance = 1 -> the neighbors farthest apart are adjacent pixels,
				# so there is actually only one side to this branch, go with any of those two neighbors and ditch the other
				if max_sqdistance == 1:
					remainder[neighbors[best_neighbor2][0], neighbors[best_neighbor2][1]] = 0
					base_y = neighbors[best_neighbor1][0]
					base_x = neighbors[best_neighbor1][1]
				# Otherwise, we need to go both ways. To do that, call _extract_branch recursively with a temporary deque,
				# and an initial position to make sure it goes in the right direction (see the next part of the function)
				else:
					subbranch.clear()
					_extract_branch(remainder, subbranch, neighbors[best_neighbor1][0], neighbors[best_neighbor1][1], base_y, base_x)
					
					# Then add it in reverse at the beginning of our branch, and continue this loop for the other direction
					it = subbranch.begin()
					while it != subbranch.end():
						branch.push_front(deref(it))
						postinc(it)
					base_y = neighbors[best_neighbor2][0]
					base_x = neighbors[best_neighbor2][1]
			# There is a previous position, so we just need to select the one neighbor that is farthest from it
			else:
				# First point and initial position, we are in a recursive call and there are multiple neighbor,
				# select the neighbor farthest from the initial point of the parent branch to make sure it goes in the right direction
				if branch.size() == 1 and (init_y >= 0 and init_x >= 0):
					last_point = (init_y, init_x)
				# Just randomly found multiple neighbors on the curve, the previous position is just the previous point
				else:
					last_point = branch.back()
				
				# Maximize the distance to the previous position and find the farthest neighbor
				max_sqdistance = 0
				best_neighbor1 = -1
				for i in range(neighbor_index):
					sqdistance = (neighbors[i][0] - last_point[0])**2 + (neighbors[i][1] - last_point[1])**2
					if sqdistance > max_sqdistance:
						max_sqdistance = sqdistance
						best_neighbor1 = i
				# Delete the other neighbors completely, otherwise they could parasite the following loops
				# Those are just line overthickness, if it is actually a branching curve, it will just
				# clear the first pixel and be caught back later
				for i in range(neighbor_index):
					if i != best_neighbor1:
						remainder[neighbors[i][0], neighbors[i][1]] = 0
				
				# And go to the next one
				base_y = neighbors[best_neighbor1][0]
				base_x = neighbors[best_neighbor1][1]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cpdef list extract_branches(uint8_t[:, :] image, size_t min_size=1):
	"""Extract all "branches", continuous, 1-pixel wide discrete curves in a binary image
	   The points within a branch are in order, with no discontinuity
	   - image    : uint8_t[y, x]       : Binary image in which to extract curves
	   - min_size : size_t              : Minimal number of pixels in a branch to keep it
	<-------------- list<ndarray[2, N]> : List of discrete curves in (x, y)ᵀ integer pixel coordinates
	"""
	cdef uint8_t* remainder_ptr = <uint8_t*> malloc(image.shape[0] * image.shape[1] * sizeof(uint8_t))
	cdef uint8_t[:, ::1] remainder = <uint8_t[:image.shape[0], :image.shape[1]:1]> remainder_ptr
	cdef deque[(Py_ssize_t, Py_ssize_t)] branch_build
	cdef long[:, ::1] branch_view
	cdef Py_ssize_t start_branches_count = -1
	cdef list branches = []
	cdef Py_ssize_t y, x
	cdef size_t i

	# We need a state mask to keep track of which points remain to be handled
	# `remainder` is our state mask, 0 = to ignore (originally black pixel, or already handled),
	# 1 = to be handled
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			if image[y, x] > 0:
				remainder[y, x] = 1
			else:
				remainder[y, x] = 0
	
	# No loop on the remainder, and extract a branch whenever we find an interesting point
	# Normally, all curve points are handled in one go
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			# Nothing to see here
			if remainder[y, x] == 0:
				continue

			# Check the amount of 8-neighbors this point has to know what to do
			num_neighbors = _neighbor_count(remainder, y, x)

			# Isolated point -> just discard it
			if num_neighbors == 0:
				remainder[y, x] = 0
			# Has neighbors -> extract a branch
			elif num_neighbors >= 1:
				branch_build.clear()
				_extract_branch(remainder, branch_build, y, x, -1, -1)
				if branch_build.size() >= min_size:
					# It is in a C++ deque, transfer the points in a fixed numpy array
					branch = np.empty((2, branch_build.size()), dtype=int)
					branch_view = branch
					for i in range(branch_build.size()):
						branch_view[0, i] = branch_build[i][1]
						branch_view[1, i] = branch_build[i][0]
					branches.append(branch)
	
	free(remainder_ptr)
	return branches

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cpdef list cut_line_angles(line, int filter_size, double filter_deviation, int min_branch_size, double min_branch_length, double max_curvature):
	"""Split a discrete curve where its curvature gets too high.
	   The curvature is calculated locally in rad/length unit, and smoothed using a gaussian filter
	   Here the measures are in "length units", this depends on the curve unit (pixels, meters, …)
	   - line              : ndarray[2, N]       : Discrete curve to split
	   - filter_size       : int                 : Size of the gaussian kernel
	   - filter_deviation  : double              : Standard deviation σ of the gaussian filter
	   - min_branch_size   : int                 : Only keep curve sections that have more points than `min_branch_size`
	   - min_branch_length : double              : Only keep curve sections that are longer that `min_branch_length` length units
	   - max_curvature     : double              : Split the curve when its curvature gets over this threshold in rad/length unit
	<----------------------- list<ndarray[2, N]> : List of resulting curve sections
	"""
	cdef list cluster_lines = []
	cdef Py_ssize_t section_start, section_end, i, j, curvature_index = 0
	cdef double[:, :] line_view = line

	cdef double* curvature_buffer = <double*>malloc(filter_size * sizeof(double))
	cdef double* gaussian_filter = <double*>malloc(filter_size * sizeof(double))
	cdef double gaussian_weight = 0, smooth_curvature, curvature, previous_curvature = -1
	cdef double prev_vector_x, prev_vector_y, next_vector_x, next_vector_y, prev_vector_norm, next_vector_norm
	
	# Precompute the gaussian filter kernel
	for j in range(filter_size):
		gaussian_filter[j] = exp(-(j-filter_size//2)**2 / (2*filter_deviation**2))
		gaussian_weight += gaussian_filter[j]
	# Normalize it such that its sum is 1
	for j in range(filter_size):
		gaussian_filter[j] /= gaussian_weight
		
	# Now loop on the curve, computing the pointwise curvature, filtering it, and splitting the curve when necessary
	section_start = 0  # Start index of the section (first index to have a low enough curvature)
	section_end = 0    # End index of the section (first index to have a curvature too high)
	# No curvature calculation on the extreme point, we need both adjacent vectors
	for i in range(1, line_view.shape[1]-1):
		# Calculate the previous and next vectors around this point
		prev_vector_x = line_view[0, i] - line_view[0, i-1]
		prev_vector_y = line_view[1, i] - line_view[1, i-1]
		next_vector_x = line_view[0, i+1] - line_view[0, i]
		next_vector_y = line_view[1, i+1] - line_view[1, i]
		prev_vector_norm = sqrt(prev_vector_x*prev_vector_x + prev_vector_y*prev_vector_y)
		next_vector_norm = sqrt(next_vector_x*next_vector_x + next_vector_y*next_vector_y)

		# Now, the curvature is the difference in angle between those two vectors, divided by the local mean length of a discrete curve segment
		curvature = _acos_clip((prev_vector_x*next_vector_x + prev_vector_y*next_vector_y) / (prev_vector_norm * next_vector_norm)) / ((prev_vector_norm + next_vector_norm) / 2)
		
		# At first, initialize the buffer with this curvature value all over
		if i == 1:
			for j in range(filter_size):
				curvature_buffer[j] = curvature
		
		curvature_buffer[curvature_index] = curvature
		
		# To keep the previous curvatures up to `filter_size` values, we need a queue
		# But as it’s fixed-size, no need to use an expensive structure, just use an array and cycle over it
		# with a modulo to keep track of the order of arrival.
		# WARNING : This requires Python-style modulo (or Real™ modulo), that makes negative % positive cycle back to positive,
		#           unlike C-style or JS-style *remainder* operator that makes a modulo like if the number was positive but shoves a minus sign in front of it
		smooth_curvature = 0
		for j in range(filter_size):
			smooth_curvature += gaussian_filter[j]*curvature_buffer[(curvature_index - (filter_size - j - 1)) % filter_size]
		
		# Increment-and-cycle-back the index in the queue buffer for next time
		curvature_index = (curvature_index + 1) % filter_size

		# Now that we have the gaussian-filtered curvature, check if the too-high status has gone high-to-low (start a new section)
		# or low-to-high (end of the section)
		# The indices are `- filter_size//2` because `smooth_curvature` is the result of the gaussian filter around the value
		# at the middle of the filter, that is `filter_size//2` points backwards
		if previous_curvature >= 0:
			# Was bad, is now ok -> start a new section
			if (smooth_curvature < max_curvature) and (previous_curvature >= max_curvature):
				section_start = i - filter_size//2
			# Was ok, now bad -> end the section, split the curve and add the section to the list
			elif (smooth_curvature >= max_curvature) and (previous_curvature < max_curvature):
				section_end = i - filter_size//2
				# Check the size and length of the section
				if section_end - section_start > min_branch_size + 1:
					section = line[:, section_start:section_end]
					if trajeometry.line_length(section) > min_branch_length:
						cluster_lines.append(section)
		previous_curvature = smooth_curvature
	
	# As the curvature check is always `filter_size//2` back, we need to check the last few points
	# with the last local curvature value. This isn’t critical, but it helps eliminating weird angles at the end of curves
	for i in range(filter_size // 2):
		curvature_buffer[curvature_index] = curvature
		smooth_curvature = 0
		for j in range(filter_size):
			smooth_curvature += gaussian_filter[j]*curvature_buffer[(curvature_index - (filter_size - j - 1)) % filter_size]
		curvature_index = (curvature_index + 1) % filter_size
		if previous_curvature >= 0:
			if (smooth_curvature < max_curvature) and (previous_curvature >= max_curvature):
				section_start = line_view.shape[1] - filter_size//2 + i
			elif (smooth_curvature >= max_curvature) and (previous_curvature < max_curvature):
				section_end = line_view.shape[1] - filter_size//2 + i
				if section_end - section_start > min_branch_size + 1:
					section = line[:, section_start:section_end]
					if trajeometry.line_length(section) > min_branch_length:
						cluster_lines.append(section)
		previous_curvature = smooth_curvature

	# Now we need to check a few things to make sure not to forget some sections
	# First, at this point, if `section_start` is further than `section_end`,
	# a sections has been started but has remained valid until the end, so we need to add it as is till the end
	if section_start > section_end:
		if line_view.shape[1] - section_start > min_branch_size + 1:
			section = line[:, section_start:]
			if trajeometry.line_length(section) > min_branch_length:
				cluster_lines.append(section)
	# And if both start and end are still 0, no splits have taken place at all, so add the whole curve
	elif section_start == 0 and section_end == 0:
		if line_view.shape[1] > min_branch_size + 1 and trajeometry.line_length(line) > min_branch_length:
			cluster_lines.append(line)
	
	free(gaussian_filter)
	free(curvature_buffer)

	return cluster_lines

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cpdef list filter_lines(list lines, int savgol_degree=2, int initial_filter_window=20, int smoothing_filter_window=12, double branch_step=1, double min_branch_length=5, double min_line_length=5, double max_curvature=1, int curvature_filter_size=7, double curvature_filter_deviation=1, double merge_max_distance=100, double angle_diff_threshold=np.pi/3, double merge_score_threshold=50):
	"""Take a list of discrete curves, smooth them, split them at angles and merge those that would make continuous curves,
	   such that the result is a set of smooth discrete curves
	   - lines                      : list<ndarray[2, N]> : List of discrete curves
	   - savgol_degree              : int                 : Polynomial degree of the Savitzky-Golay filters used to smooth curves
	   - initial_filter_window      : int                 : Size of the window of the Savitzky-Golay filter applied before doing anything else
	   - smoothing_filter_window    : int                 : Size of the final Savitzky-Golay filter used to smooth the resulting curves
	   - branch_step                : double              : Curves are resampled to have point at this distance from one another
	   - min_branch_length          : double              : Minimum length (in curve length units) of intermediate curve sections, shorter sections are eliminated
	   - min_line_length            : double              : Minimum length of the resulting curves, shorter curves are eliminated
	   - max_curvature              : double              : Curves are split when their curvature in rad/unit exceeds this threshold
	   - curvature_filter_size      : int                 : Size of the gaussian filter used to smooth the curvature in `cut_line_angles`
	   - curvature_filter_deviation : double              : Standard deviation σ of the gaussian filter used to smooth the curvature in `cut_line_angles`
	   - merge_max_distance         : double              : Maximum distance at which two curves can be merged together
	   - angle_diff_threshold       : double              : Maximum relative angle in radians at which two curves can be merged together
	   - merge_score_threshold      : double              : Maximum score at which two curves are considered candidate for merge
	<-------------------------------- list<ndarray[2, M]> : List of resulting discrete curves. There are no guarantees of size, number of curves or any correspondance whatsoever with the initial curves
	"""
	
	# Filter the extracted branches, and cut them where the angle gets too sharp, to avoid rough edges getting in the way of merging
	cdef list cluster_lines = []
	for line in lines:
		if line.shape[1] > savgol_degree + 1 and trajeometry.line_length(line) > min_branch_length:
			# Smooth the branch, at this points it is contiguous pixel coordinates
			filtered_branch = trajeometry.savgol_filter(line, trajeometry.savgol_window(initial_filter_window, line.shape[1]), savgol_degree)
			resampled_branch = trajeometry.resample_curve(filtered_branch, branch_step)

			if resampled_branch.shape[1] > savgol_degree + 1:
				cluster_lines.extend(cut_line_angles(resampled_branch, curvature_filter_size, curvature_filter_deviation, savgol_degree, min_branch_length, max_curvature))
	
	# Merge lines that are in the continuity of each other
	# This checks all the combinations of line and merge them when possible
	
	# This is a cache of combinations already checked, to avoid repeating work
	# All this is repeated until no more lines are deemed similar enough to be merged together
	# Cython ctuples are great but have no == operator, so we’re forced to use C++ pairs
	cdef double[:, :] line1, line2
	cdef cset[pair[Py_ssize_t, Py_ssize_t]] passed_combinations
	cdef cset[pair[Py_ssize_t, Py_ssize_t]].iterator it
	cdef double[:, ::1] merged_line_view
	cdef Py_ssize_t i

	cdef _MergeState merge_candidate, merge_result = _MergeState(merge=False, score=-1, flip1=False, flip2=False)
	cdef Py_ssize_t index1, index2, merge_index = -1
	while True:
		for index1 in range(len(cluster_lines)):
			merge_result.merge = False
			line1 = cluster_lines[index1]

			for index2 in range(index1+1, len(cluster_lines)):
				if passed_combinations.find(pair[Py_ssize_t, Py_ssize_t](index1, index2)) != passed_combinations.end():
					continue  # Already done, skip

				line2 = cluster_lines[index2]
				merge_candidate = merge_lines(line1, line2, merge_max_distance=merge_max_distance, angle_diff_threshold=angle_diff_threshold, merge_score_threshold=merge_score_threshold)
				# Lines can be merged and they are the best candidates yet : set them as the current merge candidate
				if merge_candidate.merge:
					if not merge_result.merge or merge_candidate.score < merge_result.score:
						merge_result = merge_candidate
						merge_index = index2

				passed_combinations.insert(pair[Py_ssize_t, Py_ssize_t](index1, index2))
			
			# There is a merge candidate : merge the lines
			if merge_result.merge:
				# Remove the old lines and add the new, merged one
				line2 = cluster_lines[merge_index]
				cluster_lines.pop(max(index1, merge_index))
				cluster_lines.pop(min(index1, merge_index))

				merged_line = np.empty((2, line1.shape[1] + line2.shape[1]))
				merged_line_view = merged_line

				for i in range(line1.shape[1]):
					if merge_result.flip1:
						merged_line_view[:, i] = line1[:, line1.shape[1] - i - 1]
					else:
						merged_line_view[:, i] = line1[:, i]
				for i in range(line2.shape[1]):
					if merge_result.flip2:
						merged_line_view[:, line1.shape[1] + i] = line2[:, line2.shape[1] - i - 1]
					else:
						merged_line_view[:, line1.shape[1] + i] = line2[:, i]
				
				cluster_lines.append(merged_line)

				# Remove the combinations that include the merged lines, to check them again
				# This should not be necessary, except for candidates that have been put away before and may be able to merge now
				# Also update the indices in the other combinations, as the merged lines have been removed
				it = passed_combinations.begin()
				while it != passed_combinations.end():
					combination = deref(it)
					if combination.first == index1 or combination.second == index1 or combination.first == merge_index or combination.second == merge_index:
						it = passed_combinations.erase(it)
					else:
						# Decrement the indices to account for the removal of lines at index `index1` and `merge_index`
						# Decrement by 2 if it was after both lines, by 1 if it was after only one of them
						if combination.first > index1 and combination.first > merge_index:
							deref(it).first -= 2
						elif combination.first > index1 or combination.first > merge_index:
							deref(it).first -= 1
						
						if combination.second > index1 and combination.second > merge_index:
							deref(it).second -= 2
						elif combination.second > index1 or combination.second > merge_index:
							deref(it).second -= 1
						postinc(it)
				break  # Break the current loop and start over with the merged lines
		else:
			break  # There have been no breaks in all this process -> quit, no more merges possible
	
	# Smooth and filter the resulting lines, eliminate those that might be too short
	cdef list final_lines = []
	for line in cluster_lines:
		if trajeometry.line_length(line) > min_line_length:
			resampled_line = trajeometry.resample_curve(line, branch_step)
			filtered_line = trajeometry.savgol_filter(resampled_line, trajeometry.savgol_window(smoothing_filter_window, resampled_line.shape[1]), savgol_degree)
			final_lines.append(filtered_line)
	
	return final_lines