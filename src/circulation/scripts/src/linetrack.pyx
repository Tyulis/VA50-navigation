# distutils: language=c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False

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
Module that does curve extraction from binary images
"""

import numpy as np

import trajeometry

cimport cython
cimport trajutil
cimport trajeometry
cimport numpy as cnp
from cython.operator cimport dereference as deref, postincrement as postinc
from libc.math cimport sqrt, acos, exp, M_PI, INFINITY, NAN, cos, sin
from libc.stdlib cimport malloc, free
from libcpp.deque cimport deque
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.utility cimport pair
from numpy cimport uint8_t


cdef extern from "<algorithm>" namespace "std":
	T max[T](T a, T b)
	T min[T](T a, T b)


# Result of a joining capability check
cdef struct _MergeState:
	bint merge        # True => the lines can be merged, False => they don’t
	bint flip1        # Whether to flip the first line before concatenation
	bint flip2        # Whether to flip the second line before concatenation
	bint arc          # Whether the join is a circle arc
	double center_x   # Circle arc center
	double center_y   # Can’t make it an array without MSVC complaining
	double radius     # Circle arc radius
	double error      # Mean squared error of the merge
	double distance   # Minimal distance between the lines

# Accumulator used to solve circular regression by Kåsa’s method
cdef struct _KasaAccumulator:
	double sum_x
	double sum_xx
	double sum_xxx
	double sum_y
	double sum_yy
	double sum_yyy
	double sum_xy
	double sum_xxy
	double sum_xyy

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
	   - remainder      : uint8_t[y, x] : Binary values of the image, pixels at 0 are ignored
	   - branch         : deque<(Py_ssize_t, Py_ssize_t)> : Branch to fill with discrete curve points, in order
	   - base_y, base_x : Py_ssize_t : Position at which to start the extraction
	   - init_y, init_x : Py_ssize_t : Position to consider as the previous point on the curve. Use only internal, set to -1 for unused.
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
			if branch.size() == 1 and init_y < 0 and init_x < 0:
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
	
	# Now loop on the remainder, and extract a branch whenever we find an interesting point
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
	cdef double gaussian_weight = 0, smooth_curvature, curvature = 0, previous_curvature = -1
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
		
		print(smooth_curvature)
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


cdef void _fit_line(_MergeState* result, double[:, :] line1, double[:, :] line2, Py_ssize_t start1, Py_ssize_t end1, Py_ssize_t start2, Py_ssize_t end2):
	"""Check mergeability by fitting a line for all points from line1[start1:end1] and line2[start2:end2]
	   - result       : _MergeState* : Merge check result to fill
	   - line1        : double[2, N] : First curve to check
	   - line2        : double[2, M] : Second curve to check
	   - start1, end1 : Py_ssize_t   : Range of indices to use for the regression in line1 (start inclusive, end exclusive)
	   - start2, end2 : Py_ssize_t   : Range of indices to use for the regression in line2 (start inclusive, end exclusive)
	"""
	# As always in this program, we use PCA instead of the usual linear regression techniques,
	# because the latter are to infer linear *functions* y(x), while we’re looking for geometric lines in the plane,
	cdef Py_ssize_t npoints = (end1 - start1) + (end2 - start2), i
	cdef double[2] eigenvalues
	cdef double[2][2] eigenvectors
	cdef double* points_ptr = <double*> malloc(2 * npoints * sizeof(double))
	cdef double[:, ::1] points = <double[:2, :npoints:1]> points_ptr

	# Copy the relevant points in a buffer
	for i in range(start1, end1):
		points[0, i - start1] = line1[0, i]
		points[1, i - start1] = line1[1, i]
	for i in range(start2, end2):
		points[0, end1 - start1 + i - start2] = line2[0, i]
		points[1, end1 - start1 + i - start2] = line2[1, i]
	
	# Compute the PCA of those points
	if not trajutil.pca_2x2(points, eigenvalues, eigenvectors):
		result.error = INFINITY
		return
		
	# Then we can just transform the points in PCA space and compute the MSE with the distances along the secondary component
	# To save us a bit of computation, we can just calculate the value along the secondary component
	# This is not very clean, but as we won’t use the original `points` again, let’s store the ev2 coordinate in points[1]
	# to avoid recalculating or allocating another buffer
	cdef double ev2_mean = 0
	for i in range(npoints):
		points[1, i] = eigenvectors[1][0] * points[0, i] + eigenvectors[1][1] * points[1, i]
		ev2_mean += points[1, i]
	ev2_mean /= npoints
	
	result.error = 0
	for i in range(npoints):
		result.error += (points[1, i] - ev2_mean)**2
	result.error = sqrt(result.error / npoints)

	free(points_ptr)

cdef void _kasa_accumulate(_KasaAccumulator* accumulator, double[:, :] line, Py_ssize_t start, Py_ssize_t end):
	"""Compute all necessary sums for the Kåsa circular regression method over line[start:end]
	   - accumulator : _KasaAccumulator* : Structure with the actual accumulators
	   - line        : double[:, :]      : Line to accumulate over
	   - start, end  : Py_ssize_t        : Index range to accumulate over (start inclusive, end exclusive)
	"""
	cdef Py_ssize_t i
	for i in range(start, end):
		# To solve the Kåsa circular regression, we need the sums of all values of x, x², x³, y, y², y³, xy, x²y and xy²
		accumulator.sum_x   += line[0, i]
		accumulator.sum_y   += line[1, i]
		accumulator.sum_xx  += line[0, i]*line[0, i]
		accumulator.sum_yy  += line[1, i]*line[1, i]
		accumulator.sum_xxx += line[0, i]*line[0, i]*line[0, i]
		accumulator.sum_yyy += line[1, i]*line[1, i]*line[1, i]
		accumulator.sum_xy  += line[0, i]*line[1, i]
		accumulator.sum_xxy += line[0, i]*line[0, i]*line[1, i]
		accumulator.sum_xyy += line[0, i]*line[1, i]*line[1, i]

cdef void _fit_arc_kasa(_MergeState* result, double[:, :] line1, double[:, :] line2, Py_ssize_t start1, Py_ssize_t end1, Py_ssize_t start2, Py_ssize_t end2):
	"""Fit a circular arc to all points in line1[start1:end1] and line2[start2:end2]
	   - result       : _MergeState* : Merge check result to fill
	   - line1        : double[2, N] : First line to check
	   - line2        : double[2, M] : Second line to check
	   - start1, end1 : Py_ssize_t   : Index range to use for the regression on line1 (start inclusive, end exclusive)
	   - start2, end2 : Py_ssize_t   : Index range to use for the regression on line2 (start inclusive, end exclusive)
	"""
	# Fit a circle arc with Kåsa’s method
	# We use this method because it is linear, and as such, much more efficient than the non-linear regression methods
	# that require heavy iterative algorithms. Even though the results are not the same, it’s perfect for our use case
	
	# Compute the sums over all relevant points on both lines
	cdef _KasaAccumulator accumulator = _KasaAccumulator(sum_x = 0, sum_y = 0, sum_xx = 0, sum_yy = 0, sum_xxx = 0, sum_yyy = 0, sum_xy = 0, sum_xxy = 0, sum_xyy = 0)
	_kasa_accumulate(&accumulator, line1, start1, end1)
	_kasa_accumulate(&accumulator, line2, start2, end2)
	
	# Intermediate results
	cdef Py_ssize_t npoints = (end2 - start2) + (end1 - start1)
	cdef double alpha = 2 * (accumulator.sum_x**2 - npoints * accumulator.sum_xx)
	cdef double beta = 2 * (accumulator.sum_x*accumulator.sum_y - npoints * accumulator.sum_xy)
	cdef double gamma = 2 * (accumulator.sum_y**2 - npoints * accumulator.sum_yy)
	cdef double delta = accumulator.sum_xx*accumulator.sum_x - npoints*accumulator.sum_xxx + accumulator.sum_x*accumulator.sum_yy - npoints*accumulator.sum_xyy
	cdef double epsilon = accumulator.sum_xx*accumulator.sum_y - npoints*accumulator.sum_yyy + accumulator.sum_y*accumulator.sum_yy - npoints*accumulator.sum_xxy

	# Center of the circle
	result.center_x = (delta*gamma - epsilon*beta) / (alpha*gamma - beta**2)
	result.center_y = (alpha*epsilon - beta*delta) / (alpha*gamma - beta**2)
	
	# Now, this method only gives us the center of the circle
	# Then the radius is just the mean of distances to the center
	result.radius = 0
	for i in range(start1, end1):
		result.radius += sqrt((line1[0, i] - result.center_x)**2 + (line1[1, i] - result.center_y)**2)
	for i in range(start2, end2):
		result.radius += sqrt((line2[0, i] - result.center_x)**2 + (line2[1, i] - result.center_y)**2)
	result.radius /= npoints

	# And finally, the RMS error with distance_to_center - radius as base error
	result.error = 0
	for i in range(start1, end1):
		result.error += (sqrt((line1[0, i] - result.center_x)**2 + (line1[1, i] - result.center_y)**2) - result.radius) ** 2
	for i in range(start2, end2):
		result.error += (sqrt((line2[0, i] - result.center_x)**2 + (line2[1, i] - result.center_y)**2) - result.radius) ** 2
	result.error = sqrt(result.error / npoints)


cdef _MergeState check_mergeability(double[:, :] line1, double[:, :] line2, Py_ssize_t estimate_start, Py_ssize_t estimate_end, double merge_max_distance, double max_angle_diff, double max_rmse):
	"""Check whether two curves can be joined, and how
	   First check whether a simple line joint can do the job, then with a circle arc
	   - line1                        : double[2, N] : First line to check
	   - line2                        : double[2, M] : Second line to check
	   - estimate_start, estimate_end : Py_ssize_t   : Range index for the joint fittings, relative to the closest points between both curves
	   - merge_max_distance           : double       : Curves with relative distance higher than this value can never be merged
	   - max_angle_diff               : double       : Curves with relative angle in radians higher than this value can never be merged
	   - max_rmse                     : double       : Curves can be merged when one of the fitting methods produces a Root Mean Squared Error lower than this threshold
	<---------------------------------- _MergeState  : Check result with all necessary information, check the structure definition
	"""
	cdef _MergeState result = _MergeState(merge=False, flip1=False, flip2=False, arc=False, center_x=NAN, center_y=NAN, radius=NAN, error=INFINITY, distance=NAN)
	cdef Py_ssize_t line1_length = line1.shape[1], line2_length = line2.shape[1]
	
	# The lines could be in any direction
	# We need the points and vectors that directly face each other
	# Those are the squared distance to save a few sqrt() calls
	cdef double extreme_distance_00 = (line1[0, 0] - line2[0, 0])**2 + (line1[1, 0] - line2[1, 0])**2
	cdef double extreme_distance_10 = (line1[0, line1_length-1] - line2[0, 0])**2 + (line1[1, line1_length-1] - line2[1, 0])**2
	cdef double extreme_distance_01 = (line1[0, 0] - line2[0, line2_length-1])**2 + (line1[1, 0] - line2[1, line2_length-1])**2
	cdef double extreme_distance_11 = (line1[0, line1_length-1] - line2[0, line2_length-1])**2 + (line1[1, line1_length-1] - line2[1, line2_length-1])**2

	# To concatenate them at the end, the lines must be like 0 --line1--> -1 |----| 0 --line2--> -1
	# So the closest points must be line1[-1] and line2[0]
	# So flip 1 if the closest point is 0, flip 2 if the closest point is -1
	if extreme_distance_00 <= extreme_distance_01 and extreme_distance_00 <= extreme_distance_10 and extreme_distance_00 <= extreme_distance_11:
		result.distance = sqrt(extreme_distance_00)
		result.flip1 = True
		result.flip2 = False
	elif extreme_distance_01 <= extreme_distance_00 and extreme_distance_01 <= extreme_distance_10 and extreme_distance_01 <= extreme_distance_11:
		result.distance = sqrt(extreme_distance_01)
		result.flip1 = True
		result.flip2 = True
	elif extreme_distance_10 <= extreme_distance_00 and extreme_distance_10 <= extreme_distance_01 and extreme_distance_10 <= extreme_distance_11:
		result.distance = sqrt(extreme_distance_10)
		result.flip1 = False
		result.flip2 = False
	else:
		result.distance = sqrt(extreme_distance_11)
		result.flip1 = False
		result.flip2 = True

	# The closest points are too far away -> ditch this combination
	if result.distance > merge_max_distance:
		return result
	
	# Get the initial points and vectors, and the estimate index ranges depending on the flip status of both curves
	cdef Py_ssize_t start1, start2, end1, end2
	cdef double[2] point1, point2, vector1, vector2, joint_vector
	if result.flip1:
		start1 = estimate_start
		end1 = estimate_end
		point1[0] = line1[0, 0]
		point1[1] = line1[1, 0]
		vector1[0] = point1[0] - line1[0, 1]
		vector1[1] = point1[1] - line1[1, 1]
	else:
		start1 = line1.shape[1] - estimate_end
		end1 = line1.shape[1] - estimate_start
		point1[0] = line1[0, line1_length - 1]
		point1[1] = line1[1, line1_length - 1]
		vector1[0] = point1[0] - line1[0, line1_length - 2]
		vector1[1] = point1[1] - line1[1, line1_length - 2]
	if result.flip2:
		start2 = line2.shape[1] - estimate_end
		end2 = line2.shape[1] - estimate_start
		point2[0] = line2[0, line2_length - 1]
		point2[1] = line2[1, line2_length - 1]
		# We need it reversed to compare its angle with the joint vector that goes 1 -> 2
		vector2[0] = -(point2[0] - line2[0, line2_length - 2])
		vector2[1] = -(point2[1] - line2[1, line2_length - 2])
	else:
		start2 = estimate_start
		end2 = estimate_end
		point2[0] = line2[0, 0]
		point2[1] = line2[1, 0]
		vector2[0] = -(point2[0] - line2[0, 1])
		vector2[1] = -(point2[1] - line2[1, 1])
	
	# Get the vector from point1 to point2, and fail if the extreme vector of one of the curves differs too much from it
	joint_vector[0] = point2[0] - point1[0]
	joint_vector[1] = point2[1] - point1[1]
	if trajeometry.vector_angle(vector1, joint_vector) > max_angle_diff or trajeometry.vector_angle(vector2, joint_vector) > max_angle_diff:
		return result
	
	# Clamp the estimate range to the amount of points
	if start1 < 0: start1 = 0
	if start2 < 0: start2 = 0
	if end1 >= line1.shape[1]: end1 = line1.shape[1]
	if end2 >= line2.shape[1]: end2 = line2.shape[1]

	# Try the line fitting
	_fit_line(&result, line1, line2, start1, end1, start2, end2)
	if result.error < max_rmse:
		result.merge = True
		result.arc = False
		return result

	# If it’s unsatisfactory, check with a circle arc
	_fit_arc_kasa(&result, line1, line2, start1, end1, start2, end2)
	if result.error < max_rmse:
		result.merge = True
		result.arc = True
	
	# Nope, nothing is satisfactory, return with .merge=False
	return result

cdef cnp.ndarray _join_curves_line(_MergeState merge_result, double[:, :] line1, double[:, :] line2):
	"""Join two curves with a line between the extreme points
	   - merge_result : _MergeState     : Merge check result on which to base the operation
	   - line1        : double[2, N]    : First line to join
	   - line2        : double[2, M]    : Second line to join
	<------------------ ndarray[2, M+N] : Resulting joined curve
	"""
	cdef cnp.ndarray merged_line = np.empty((2, line1.shape[1] + line2.shape[1]))
	cdef double[:, ::1] merged_line_view = merged_line
	cdef Py_ssize_t i

	# Just concatenate them, it will make a straight segment that will get resampled later
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
	return merged_line

cdef cnp.ndarray _join_curves_arc(_MergeState merge_result, double[:, :] line1, double[:, :] line2, double branch_step):
	"""Join two curves with a circle arc between the extreme points
	   - merge_result : _MergeState          : Merge check result with the arc parameters
	   - line1        : double[2, N]         : First curve to join
	   - line2        : double[2, M]         : Second curve to join
	   - branch_step  : double               : Base distance between curve points for the interpolation
	<------------------ double[2, N + x + M] : Joined and interpolated curve"""
	# Get the relevant extreme points
	cdef double[2] last_point1, last_point2
	if merge_result.flip1:
		last_point1[0] = line1[0, 0]
		last_point1[1] = line1[1, 0]
	else:
		last_point1[0] = line1[0, line1.shape[1] - 1]
		last_point1[1] = line1[1, line1.shape[1] - 1]
	if merge_result.flip2:
		last_point2[0] = line2[0, line2.shape[1] - 1]
		last_point2[1] = line2[1, line2.shape[1] - 1]
	else:
		last_point2[0] = line2[0, 0]
		last_point2[1] = line2[1, 0]
	
	# Compute the vectors from the center of the circle to those extreme points
	# The interpolation arc is over the relative angle between those two vectors
	cdef double[2] center_vector1, center_vector2 
	center_vector1[0] = last_point1[0] - merge_result.center_x
	center_vector1[1] = last_point1[1] - merge_result.center_y
	center_vector2[0] = last_point2[0] - merge_result.center_x
	center_vector2[1] = last_point2[1] - merge_result.center_y
	cdef double joint_angle = trajeometry.vector_angle(center_vector1, center_vector2)

	# branch_step / radius is the angle step around that circle in radians
	# N angle steps -> N - 1 intermediate points
	cdef double angle_step = branch_step / merge_result.radius
	cdef Py_ssize_t i, joint_points = <Py_ssize_t> (joint_angle / angle_step - 1)
	
	cdef cnp.ndarray merged_line = np.empty((2, line1.shape[1] + joint_points + line2.shape[1]))
	cdef double[:, ::1] merged_line_view = merged_line

	# Copy the original lines at their respective positions
	for i in range(line1.shape[1]):
		if merge_result.flip1:
			merged_line_view[:, i] = line1[:, line1.shape[1] - i - 1]
		else:
			merged_line_view[:, i] = line1[:, i]
	for i in range(line2.shape[1]):
		if merge_result.flip2:
			merged_line_view[:, line1.shape[1] + joint_points + i] = line2[:, line2.shape[1] - i - 1]
		else:
			merged_line_view[:, line1.shape[1] + joint_points + i] = line2[:, i]
	
	# Then interpolate between the center vectors
	# The joint points are just the center vectors rotated by the angle step, and with their lengths interpolated
	cdef double center_vector1_norm = sqrt(center_vector1[0]*center_vector1[0] + center_vector1[1]*center_vector1[1])
	cdef double center_vector2_norm = sqrt(center_vector2[0]*center_vector2[0] + center_vector2[1]*center_vector2[1])
	cdef double rotation_angle, length_factor
	for i in range(joint_points):
		rotation_angle = angle_step * (i + 1)
		length_factor = (rotation_angle*center_vector2_norm + (joint_angle - rotation_angle)*center_vector1_norm) / (joint_angle * center_vector1_norm)
		merged_line_view[0, line1.shape[1] + i] = (cos(rotation_angle)*center_vector1[0] - sin(rotation_angle)*center_vector1[1]) * length_factor + merge_result.center_x
		merged_line_view[1, line1.shape[1] + i] = (sin(rotation_angle)*center_vector1[0] + cos(rotation_angle)*center_vector1[1]) * length_factor + merge_result.center_y
	return merged_line

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cpdef list filter_lines(list lines, int savgol_degree=2, int initial_filter_window=20, int smoothing_filter_window=12, double branch_step=1, double min_branch_length=5, double min_line_length=5, double max_curvature=1, int curvature_filter_size=7, double curvature_filter_deviation=1, double merge_max_distance=100, Py_ssize_t estimate_start=2, Py_ssize_t estimate_end=8, double max_angle_diff=1, double max_rmse=1):
	"""Take a list of discrete curves, smooth them, split them at angles and merge those that would make continuous curves,
	   such that the result is a set of smooth discrete curves
	   - lines                        : list<ndarray[2, N]> : List of discrete curves
	   - savgol_degree                : int                 : Polynomial degree of the Savitzky-Golay filters used to smooth curves
	   - initial_filter_window        : int                 : Size of the window of the Savitzky-Golay filter applied before doing anything else
	   - smoothing_filter_window      : int                 : Size of the final Savitzky-Golay filter used to smooth the resulting curves
	   - branch_step                  : double              : Curves are resampled to have point at this distance from one another
	   - min_branch_length            : double              : Minimum length (in curve length units) of intermediate curve sections, shorter sections are eliminated
	   - min_line_length              : double              : Minimum length of the resulting curves, shorter curves are eliminated
	   - max_curvature                : double              : Curves are split when their curvature in rad/unit exceeds this threshold
	   - curvature_filter_size        : int                 : Size of the gaussian filter used to smooth the curvature in `cut_line_angles`
	   - curvature_filter_deviation   : double              : Standard deviation σ of the gaussian filter used to smooth the curvature in `cut_line_angles`
	   - merge_max_distance           : double              : Maximum distance at which two curves can be merged together
	   - estimate_start, estimate_end : Py_ssize_t          : Range index for the joint fittings, relative to the closest points between both curves
	   - max_angle_diff               : double              : Maximum relative angle in radians at which two curves can be merged together
	   - merge_score_threshold        : double              : Maximum score at which two curves are considered candidate for merge
	   - max_rmse                     : double              : Maximum value of the Root Mean Squared Error with the fitted estimator to accept a merge
	<---------------------------------- list<ndarray[2, M]> : List of resulting discrete curves. There are no guarantees of size, number of curves or any correspondance whatsoever with the initial curves
	"""
	
	# Filter the extracted branches, and cut them where the angle gets too sharp, to avoid rough edges getting in the way of merging
	cdef Py_ssize_t i
	cdef list cut_lines = []
	for line in lines:
		if line.shape[1] > savgol_degree + 1 and trajeometry.line_length(line) > min_branch_length:
			# Smooth the branch, at this points it is contiguous pixel coordinates
			resampled_branch = trajeometry.resample_curve(line, branch_step)
			if resampled_branch.shape[1] <= savgol_degree + 1:
				continue
			
			cut_lines.extend(cut_line_angles(resampled_branch, curvature_filter_size, curvature_filter_deviation, savgol_degree, min_branch_length, max_curvature))
	
	cdef list cluster_lines = []
	for i, line in enumerate(cut_lines):
		if line.shape[1] > savgol_degree + 1:
			filtered_branch = trajeometry.savgol_filter(line, trajeometry.savgol_window(initial_filter_window, line.shape[1]), savgol_degree)
			if filtered_branch.shape[1] > savgol_degree + 1:
				cluster_lines.append(filtered_branch)
				

	# Merge lines that are in the continuity of each other
	# This checks all the combinations of line and merge them when possible
	
	# This is a cache of combinations already checked, to avoid repeating work
	# All this is repeated until no more lines are deemed similar enough to be merged together
	# Cython ctuples are great but have no == operator, so we’re forced to use C++ pairs
	cdef double[:, :] line1, line2
	cdef cset[pair[Py_ssize_t, Py_ssize_t]] passed_combinations
	cdef cset[pair[Py_ssize_t, Py_ssize_t]].iterator it

	cdef _MergeState merge_candidate, merge_result = _MergeState(merge=False, flip1=False, flip2=False, arc=False, center_x=NAN, center_y=NAN, radius=NAN, error=INFINITY, distance=NAN)
	cdef Py_ssize_t index1, index2, merge_index = -1
	while True:
		for index1 in range(len(cluster_lines)):
			merge_result.merge = False
			line1 = cluster_lines[index1]

			for index2 in range(index1+1, len(cluster_lines)):
				if passed_combinations.find(pair[Py_ssize_t, Py_ssize_t](index1, index2)) != passed_combinations.end():
					continue  # Already done, skip

				line2 = cluster_lines[index2]
				merge_candidate = check_mergeability(line1, line2, estimate_start, estimate_end, merge_max_distance, max_angle_diff, max_rmse)
				# Lines can be merged and they are the best candidates yet : set them as the current merge candidate
				if merge_candidate.merge:
					if not merge_result.merge or merge_candidate.error*merge_candidate.distance < merge_result.error*merge_result.distance:
						merge_result = merge_candidate
						merge_index = index2

				passed_combinations.insert(pair[Py_ssize_t, Py_ssize_t](index1, index2))
			
			# There is a merge candidate : merge the lines
			if merge_result.merge:
				# Remove the old lines and add the new, merged one
				line2 = cluster_lines[merge_index]
				cluster_lines.pop(max(index1, merge_index))
				cluster_lines.pop(min(index1, merge_index))

				# Join the curve with the right joint type
				if merge_result.arc:
					merged_line = _join_curves_arc(merge_result, line1, line2, branch_step)
				else:
					merged_line = _join_curves_line(merge_result, line1, line2)
								
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