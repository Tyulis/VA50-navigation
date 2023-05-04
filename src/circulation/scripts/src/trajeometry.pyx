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
Module that does all the basic discrete curve geometry for the rest of the program
"""

import numpy as np

cimport cython
from libc.math cimport sqrt, acos, INFINITY, NAN, M_PI
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

# Return value of _project_on_curve_index
cdef struct _ProjectionIndex:
	int index
	double vector_factor

# Return value of _next_point
cdef struct _NextPoint:
	double x, y
	int index
	double vector_factor

cdef double _acos_clip(double val) noexcept nogil:
	"""Utility function, when we try to get the angle between two vectors using the dot product,
	   sometimes floating-points shenanigans make the cos value get a little bit outside [-1, 1],
	   so this puts it back into the right range and computes the arccos"""
	if val < -1:
		return acos(-1)
	elif val > 1:
		return acos(1)
	else:
		return acos(val)


cpdef double vector_angle(double[:] v1, double[:] v2):
	"""Compute the relative angle between two n-dimensional vectors
	   - v1 : double[D] : First vector
	   - v2 : double[D] : Second vector
	<-------- double    : Relative angle in radians
	"""
	# Use the usual dot product formula, arccos(v₁·v₂ / (||v₁|| × ||v₂||))
	cdef double dotproduct = 0, v1_norm = 0, v2_norm = 0
	cdef Py_ssize_t i
	for i in range(v1.shape[0]):
		dotproduct += v1[i] * v2[i]
		v1_norm += v1[i] * v1[i]
		v2_norm += v2[i] * v2[i]
	return _acos_clip(dotproduct / (sqrt(v1_norm)*sqrt(v2_norm)))


cpdef double line_length(double[:, :] curve):
	"""Compute the actual length of a discrete curve, as the sum of the lengths of its segments
	   - curve : double[2, N] : Discrete curve to compute the length of
	<----------- double       : Length of the curve in pixels
	"""
	cdef double length = 0
	cdef Py_ssize_t i, num_points = curve.shape[1]
	for i in range(num_points - 1):
		length += sqrt((curve[0, i] - curve[0, i+1])**2 + (curve[1, i] - curve[1, i+1])**2)
	return length


cdef _ProjectionIndex _project_on_curve_index(double point_x, double point_y, double[:, :] curve, bint extend, Py_ssize_t min_index):
	"""Return the closest point on the curve to the given point, including in-between curve samples
	   - point_x   : double           : Point to project, X coordinate
	   - point_y   : double           : Point to project, Y coordinate
	   - curve     : double[2, N]     : Numerical curve to project (point_x, point_y) onto
	   - extend    : bool             : If True, when the projection is before the first sample of the curve, extend it
	   - min_index : int              : Minimum index to project to
	<--------------- _ProjectionIndex : .index is the index of the closest sample to the projection in the curve
	                                  .vector_factor is the portion [0, 1] of the vector from curve[:, .index] to curve[:, .index+1] to add to the point at .index to get the actual closest point
	                                  If the projection is beyond the last sample of the curve, .index is set to -1 and .vector_factor to NAN
	                                  If the projection is before the first sample of the curve and `extend` is False, .index is set to -1 and .vector_factor to NAN
	                                  If the projection is before the first sample of the curve and `extend` is True, .vector_factor is negative and possibly below -1.
	                                  Otherwise, .vector_factor is positive and between 0 and 1
	"""

	cdef Py_ssize_t i, best_index = -1
	cdef bint best_orthogonal = False
	cdef double best_factor = NAN, best_sqdistance = INFINITY
	cdef double current_point_x, current_point_y, vector_x, vector_y, sqdistance, factor

	for i in range(min_index, curve.shape[1]-1):
		current_point_x = curve[0, i]
		current_point_y = curve[1, i]
		vector_x = curve[0, i+1] - current_point_x
		vector_y = curve[1, i+1] - current_point_y
		
		# Find the proportion of the vector at which the point orthogonally projects
		# If it is within [0, 1[, then it can be better projected at a point on that vector
		# Otherwise we continue with the curve sample alone
		factor = ((point_x - current_point_x) * vector_x + (point_y - current_point_y) * vector_y) / (vector_x*vector_x + vector_y*vector_y)
		if (factor >= 0 or (i == 0 and extend)) and factor < 1:
			current_point_x += vector_x*factor
			current_point_y += vector_y*factor
		else:
			factor = 0
		
		# Take the square distance (to spare a square root, it’s monotonous anyway) from the point to project to the current point or the projection onto the current vector
		# The overall goal is to minimize this distance
		sqdistance = (point_x - current_point_x)**2 + (point_y - current_point_y)**2
		if sqdistance < best_sqdistance:
			best_index = i
			best_factor = factor
			best_sqdistance = sqdistance
			best_orthogonal = factor > 0
	
	# Also check the distance to the last curve sample
	# If it is closest to the point to project, then the point is beyond the curve and we fail
	# However, this might be falsified by high curvatures near the end, so if the best point found previously
	# was really orthogonal, keep it instead
	current_point_x = curve[0, curve.shape[1]-1]
	current_point_y = curve[1, curve.shape[1]-1]
	sqdistance = (point_x - current_point_x)**2 + (point_y - current_point_y)**2
	if sqdistance < best_sqdistance and not best_orthogonal:
		return _ProjectionIndex(index=-1, vector_factor=NAN)
	
	# If the point is before the first sample and no extension was required, fail
	if best_index < 0 or (best_index == 0 and best_factor <= 0 and not extend):
		return _ProjectionIndex(index=-1, vector_factor=NAN)
	else:
		return _ProjectionIndex(index=best_index, vector_factor=best_factor)

def project_on_curve_index(double[:] point, double[:, :] curve, bint extend=False, Py_ssize_t min_index=0):
	"""Return the closest point on the curve to the given point, including in-between curve samples
	   - point_x   : double[2]    : Point to project
	   - curve     : double[2, N] : Numerical curve to project `point` onto
	   - extend    : bool         : If True, when the projection is before the first sample of the curve, extend it
	   - min_index : Py_ssize_t   : Do not try to project before that index on the curve
	<--------------- int          : Index of the closest sample on the curve
	<--------------- double       : Portion [0, 1] of the vector from curve[:, index] to curve[:, index+1] to add to the point at `index` to get the actual closest point
	                                If the projection is before the first sample of the curve and `extend` is True, `vector_factor` is negative and possibly below -1.
	                                If the projection is beyond the last sample or before the first sample and `extend` is False, both return values are None
	"""
	cdef _ProjectionIndex result = _project_on_curve_index(point[0], point[1], curve, extend, min_index)
	if result.index < 0:
		return None, None
	else:
		return result.index, result.vector_factor

def project_on_curve_array(double[:] point, double[:, :] curve, bint extend=False, Py_ssize_t min_index=0):
	"""Find the index closest to the orthogonal projection of a point onto a curve defined by a point sequence
	   - point     : double[2]    : 2D point to project on the curve
	   - curve     : double[2, N] : Curve to project the point onto, as a point sequence
	   - min_index : Py_ssize_t   : Do not try to project before that index on the curve
	<--------------- int          : Index in `curve` closest to the orthogonal projection of `point`
	<--------------- double[2]    : Projected point on the curve
	                                If the projection is beyond the last sample or before the first sample and `extend` is False, both return values are None
	"""
	cdef _ProjectionIndex projection = _project_on_curve_index(point[0], point[1], curve, extend, min_index)
	if projection.index < 0:
		return None, None

	result = np.asarray(((1 - projection.vector_factor) * curve[0, projection.index] + projection.vector_factor*curve[0, projection.index + 1],
		                 (1 - projection.vector_factor) * curve[1, projection.index] + projection.vector_factor*curve[1, projection.index + 1]))
	return projection.index, result


cdef _NextPoint _next_point(Py_ssize_t closest_index, double vector_factor, double[:, :] curve, double target_distance):
	"""Find a point along the curve that is `target_distance` further than the projection of (point_x, point_y) on the curve, along its given order
	   - closest_index   : Py_ssize_t   : Index of the initial point on the curve
	   - vector_factor   : double       : Portion of the outgoing vector of the given index on the curve to add to get the initial point
	   - curve           : double[2, N] : Numerical curve to follow
	   - target_distance : double       : Distance from the projection of the initial point to the result
	<--------------------- _NextPoint   : .x and .y are the coordinates of the resulting point
	                                      .index is the index of the sample immediately before the resulting point
	                                      .vector_factor is the portion [0, 1] of the vector from curve[:, .index] to curve[:, .index+1]
										  to add to the point at .index to get the actual resulting point, might be negative if the initial vector_factor was too
	                                      If the resulting point is beyond the last sample of the curve, .index is set to -1 and .x, .y and .vector_factor are set to NAN
	"""
	cdef Py_ssize_t current_index = closest_index

	cdef Py_ssize_t curve_size = curve.shape[1]
	cdef double distance = 0
	cdef double x_derivative, y_derivative, current_point_x, current_point_y, current_vector_norm, remaining_x, remaining_y, remaining_norm
	
	# Now loop on the curve samples starting from the initial projection until the cumulative distance adds up to `target_distance` or it overshoots the curve
	while current_index < curve_size - 1:
		# Get the vector from the current sample to the next
		x_derivative = curve[0, current_index + 1] - curve[0, current_index]
		y_derivative = curve[1, current_index + 1] - curve[1, current_index]

		# When the vector_factor is non-null, take the part of the vector that remains *after* the vector_factor
		# To start at the actual initial projection, including its vector part
		remaining_x = (1 - vector_factor) * x_derivative
		remaining_y = (1 - vector_factor) * y_derivative
		remaining_norm = sqrt(remaining_x*remaining_x + remaining_y*remaining_y)
		
		# This entire vectors would make us overshoot the target : the resulting point is between this sample and the next
		# We just need to compute the necessary proportion of the vector
		if remaining_norm + distance > target_distance:
			current_vector_norm = sqrt(x_derivative*x_derivative + y_derivative*y_derivative)
			vector_factor += (target_distance - distance) / current_vector_norm
			current_point_x = curve[0, current_index] + vector_factor * x_derivative
			current_point_y = curve[1, current_index] + vector_factor * y_derivative
			return _NextPoint(x=curve[0, current_index] + vector_factor * x_derivative, y=curve[1, current_index] + vector_factor * y_derivative, index=current_index, vector_factor=vector_factor)
		
		# Otherwise, just add this distance and continue on to the next sample
		else:
			current_index += 1
			vector_factor = 0
			distance += remaining_norm

	# We have overshot the curve, fail
	return _NextPoint(x=NAN, y=NAN, index=-1, vector_factor=NAN)

def next_point(double[:] point, double[:, :] curve, double target_distance, bint extend=False, Py_ssize_t min_index=0):
	"""Find a point along the curve that is `target_distance` further than the projection of (point_x, point_y) on the curve, along its given order
	   - point           : double[2]    : Initial point
	   - curve           : double[2, N] : Numerical curve to follow
	   - target_distance : double       : Distance from the projection of the initial point to the result
	   - extend          : bool         : Stay valid even when the initial point is before the first sample of the curve
	   - min_index       : Py_ssize_t   : Do not try to start before that index on the curve
	<--------------------- double[2]    : Coordinates of the resulting point
	                                      If the initial point projection is before the first sample of the curve and `extend` is False, or if the resulting point is beyond the last sample of the curve, return None
	"""
	# First take the closest point on the curve to the initial point
	cdef _ProjectionIndex projection = _project_on_curve_index(point[0], point[1], curve, extend, min_index)
	
	# The extension shenanigans are handled entirely by _project_on_curve_index, just fail when it fails
	if projection.index < 0:
		return None

	cdef _NextPoint result = _next_point(projection.index, projection.vector_factor, curve, target_distance)
	if result.index < 0:
		return None
	else:
		return np.asarray((result.x, result.y))
	
cdef (_NextPoint, double) _next_point_score(Py_ssize_t closest_index, double vector_factor, double[:, :] curve, double[:] scores, double target_distance, bint extend, Py_ssize_t min_index):
	"""Find a point along the curve that is `target_distance` further than the projection of (point_x, point_y) on the curve, along its given order
	   - closest_index   : Py_ssize_t   : Index of the initial point on the curve
	   - vector_factor   : double       : Portion of the outgoing vector of the given index on the curve to add to get the initial point
	   - curve           : double[2, N] : Numerical curve to follow
	   - target_distance : double       : Distance from the projection of the initial point to the result
	   - extend          : bool         : Stay valid even when the initial point is before the first sample of the curve
	   - min_index       : Py_ssize_t   : Do not try to start before that index on the curve
	<--------------------- _NextPoint   : Next point parameters
	<--------------------- double       : Score of the resulting point, taking into account the surrounding samples when it is in-between two samples
	                                      If the initial point projection is before the first sample of the curve and `extend` is False, or if the resulting point is beyond the last sample of the curve, return both invalid
	"""
	cdef _NextPoint result = _next_point(closest_index, vector_factor, curve, target_distance)
	if result.index < 0:
		return (result, NAN)
	else:
		return (result, scores[result.index]*(1-result.vector_factor) + scores[result.index + 1]*result.vector_factor)

def next_point_score(double[:] point, double[:, :] curve, double[:] scores, double target_distance, bint extend=False, Py_ssize_t min_index=0):
	"""Find a point along the curve that is `target_distance` further than the projection of (point_x, point_y) on the curve, along its given order
	   - point           : double[2]    : Initial point
	   - curve           : double[2, N] : Numerical curve to follow
	   - target_distance : double       : Distance from the projection of the initial point to the result
	   - extend          : bool         : Stay valid even when the initial point is before the first sample of the curve
	   - min_index       : Py_ssize_t   : Do not try to start before that index on the curve
	<--------------------- double[2]    : Coordinates of the resulting point
	<--------------------- double       : Score of the resulting point, taking into account the surrounding samples when it is in-between two samples
	                                      If the initial point projection is before the first sample of the curve and `extend` is False, or if the resulting point is beyond the last sample of the curve, return both None
	"""
	# First take the closest point on the curve to the initial point
	cdef _ProjectionIndex projection = _project_on_curve_index(point[0], point[1], curve, extend, min_index)
	
	# The extension shenanigans are handled entirely by _project_on_curve_index, just fail when it fails
	if projection.index < 0:
		return (_NextPoint(x=NAN, y=NAN, index=-1, vector_factor=NAN), NAN)
	
	cdef _NextPoint result
	cdef double score
	
	result, score = _next_point_score(projection.index, projection.vector_factor, curve, scores, target_distance, extend, min_index)
	if result.index < 0:
		return None, None, None
	else:
		return np.asarray((result.x, result.y)), score, result.index

cpdef _ProjectionIndex project_from(double[:] point, double[:] vector, double[:, :] target_curve, bint extend, Py_ssize_t min_index):
	"""« Project » a point from the segment it belongs to, to another curve
	   The vector from the original point and its projection is orthogonal to the segment of origin, not to the target curve
	   Return the first intersection in the order of the target curve, not particularly the closest to the original point
	   - point        : double[2]        : Point to project
	   - vector       : double[2]        : The projection must be on the orthogonal vector to this relative to the point
	   - target_curve : double[2, N]     : Curve to project onto
	   - extend       : bool             : Stay valid even when the intersection is before the first sample of the curve
	   - min_index    : Py_ssize_t       : Do not search before that index on the curve
	<------------------ _ProjectionIndex : Result of the projection, as (index, interpolation)
	                                       If no valid intersection was found, return .index = -1 and .vector_factor = NAN
	"""
	cdef Py_ssize_t i
	cdef double check1_x, check1_y, check2_x, check2_y, ortho_x = -vector[1], ortho_y = vector[0]
	cdef double denominator, interp  #, factor
	for i in range(min_index, target_curve.shape[1] - 1):
		check1_x = target_curve[0, i]
		check1_y = target_curve[1, i]
		check2_x = target_curve[0, i+1]
		check2_y = target_curve[1, i+1]

		# Basically, we solve the following vector equation :
		# point + factor*ortho = (1 - interp)*check1 + interp*check2,
		# to get the intersection between the orthogonal projecting line and the curve segment line
		# If the solution is defined and the interpolation factor is in [0, 1], then the intersection is on this segment

		denominator = check1_x*ortho_y - check1_y*ortho_x - check2_x*ortho_y + check2_y*ortho_x
		# Vector and interpolation factors undefined -> the segment is colinear with the vector, skip
		if denominator == 0:
			continue

		# Actually we only need the interpolation factor
		# factor = (check1_x*check2_y - check1_x*point[1] - check1_y*check2_x + check1_y*point[0] + check2_x*point[1] - check2_y*point[0]) / denominator
		interp = (check1_x*ortho_y - check1_y*ortho_x - point[0]*ortho_y + point[1]*ortho_x) / denominator

		# Interpolation falls within the actual segment, or before if we’re in extended mode -> found the intersection
		if (interp >= 0 and interp <= 1) or (i == 0 and interp < 0):
			return _ProjectionIndex(index=i, vector_factor=interp)
	# Didn’t find any valid intersection -> fail
	return _ProjectionIndex(index=-1, vector_factor=NAN)
	

def resample_curve(double[:, :] curve, double step):
	"""Resample a curve such that points are (approximately) equidistant
	   - curve : double[2, N] : Numerical curve to resample
	   - step  : double       : Distance between the resampled points
	<----------- double[2, M] : Resampled curve. Points are *approximately* at distance `step` from each other
	                            There is no guarantee on the actual number of points in the resampled curve"""
	# Keep the same first sample
	cdef vector[(double, double)] resampled
	resampled.push_back((curve[0, 0], curve[1, 0]))

	# Then just apply _next_point in a loop until we find the end of the curve
	cdef Py_ssize_t last_index = -1
	cdef _NextPoint point = _NextPoint(x=curve[0, 0], y=curve[1, 0], index=0, vector_factor=0)
	while True:
		point = _next_point(point.index, point.vector_factor, curve, step)
		
		# Passed the end of the curve
		if point.index < 0:
			break

		last_index = point.index
		resampled.push_back((point.x, point.y))
	
	# Copy the content of the vector into a fixed numpy array
	result = np.empty((2, resampled.size()))
	cdef double[:, :] result_view = result
	cdef size_t i
	for i in range(resampled.size()):
		result[0, i] = resampled[i][0]
		result[1, i] = resampled[i][1]
	return result

cpdef bint segments_intersect(double[:] A, double[:] B, double[:] C, double[:] D):
	"""Check whether the segments AB and CD intersect with each other
	   - A : double[2] : First point of the first segment
	   - B : double[2] : Second point of the first segment
	   - C : double[2] : First point of the second segment
	   - D : double[2] : Second point of the second segment
	<------- bool      : Whether the segments AB and CD intersect
	"""
	cdef double determinant_A = (C[0]-A[0]) * (D[1]-A[1]) - (D[0]-A[0]) * (C[1]-A[1])
	cdef double determinant_B = (C[0]-B[0]) * (D[1]-B[1]) - (D[0]-B[0]) * (C[1]-B[1])
	cdef double determinant_C = (A[0]-C[0]) * (B[1]-C[1]) - (B[0]-C[0]) * (A[1]-C[1])
	cdef double determinant_D = (A[0]-D[0]) * (B[1]-D[1]) - (B[0]-D[0]) * (A[1]-D[1])

	return (((determinant_A > 0 and determinant_B < 0) or (determinant_A < 0 and determinant_B > 0)) and
	        ((determinant_C > 0 and determinant_D < 0) or (determinant_C < 0 and determinant_D > 0)))

cpdef void find_self_intersects(double[:, :] curve, bint[:] mask):
	"""Find the areas where a discrete curve intersects itself, and build a mask
	   Applying this mask to the curve remove the intersecting parts altogether
	   - curve : double[2, N] : Curve to check
	   - mask  : bool[N]      : Output for the mask
	"""
	cdef Py_ssize_t i, j, result_size = curve.shape[1]
	mask[:] = True
	# Check for intersection between each pair of segments
	# When two segments intersect, cut out everything in-between such that the points
	# before the first segment and after the second segment are linked, removing the intersecting part
	for i in range(curve.shape[1]-1):
		for j in range(i+1, curve.shape[1]-1):
			if segments_intersect(curve[:, i], curve[:, i+1], curve[:, j], curve[:, j+1]):
				mask[i+1:j+1] = False

def remove_self_intersects(curve):
	"""Remove self-intersecting parts in a discrete curve
	   - curve : ndarray[2, N] : Curve to process
	<----------- ndarray[2, M] : That same curve with self-intersecting parts removed
	                             This operation only deletes points without changing the others"""
	mask = np.ones(curve.shape[1], dtype=int)
	find_self_intersects(curve, mask)
	return curve[:, mask > 0]

def dilate_curve(double[:, :] curve, double dilation, long direction, double[:] scores=None):
	"""Apply a "dilation" operation to a discrete curve, that is, offset it regularly regardless of the curve orientation :
	   all points of the dilated curve are approximately at the same distance of their corresponding original point
	   - curve     : double[2, N] : Curve to dilate
	   - dilation  : double       : Distance of the result to the original curve
	   - direction : int          : 1 or -1, to choose the offset direction
	   - scores    : double[N]    : Scores associated to the original curve, or None.
	<--------------- double[2, M] : Dilated curve. The number of points may not be the same, as self-intersecting parts are removed
	<--------------- double[M]    : ONLY IF `scores` is not None, scores associated to each point of the dilated curve, with the same parts cut out
	"""
	# FIRST STEP : Offset the curve along its orthogonal vectors
	cdef double* dilated = <double*> malloc(curve.shape[0] * curve.shape[1] * sizeof(double))
	cdef double[:, :] dilated_view = <double[:curve.shape[0], :curve.shape[1]:1]> dilated
	cdef double[2] gradient
	cdef double gradient_norm
	cdef Py_ssize_t i
	for i in range(curve.shape[1]):
		# Calculate the gradient with [-1, 0, 1] when possible, otherwise just the outgoing or incoming vectors
		if i == 0:
			gradient[0] = curve[0, i+1] - curve[0, i]
			gradient[1] = curve[1, i+1] - curve[1, i]
		elif i == curve.shape[1] - 1:
			gradient[0] = curve[0, i] - curve[0, i-1]
			gradient[1] = curve[1, i] - curve[1, i-1]
		else:
			gradient[0] = curve[0, i+1] - curve[0, i-1]
			gradient[1] = curve[1, i+1] - curve[1, i-1]
		gradient_norm = sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1])

		# Offset by `dilation` in the direction orthogonal to the gradient, with direction to possibly reverse the orthogonal vectors
		dilated_view[0, i] = curve[0, i] - direction*gradient[1] * dilation / gradient_norm
		dilated_view[1, i] = curve[1, i] + direction*gradient[0] * dilation / gradient_norm
	
	# SECOND STEP : Remove self-intersections with the mask computed by `find_self_intersects`
	cdef bint* mask = <bint*>malloc(curve.shape[1] * sizeof(bint))
	cdef bint[::1] mask_view = <bint[:curve.shape[1]:1]> mask
	find_self_intersects(dilated_view, mask_view)
	
	cdef Py_ssize_t result_size = 0
	for i in range(curve.shape[1]):
		if mask[i]:
			result_size += 1
	
	result = np.empty((2, result_size))
	cdef double[:, :] result_view = result
	cdef Py_ssize_t result_index = 0
	for i in range(curve.shape[1]):
		if mask[i]:
			result_view[:, result_index] = dilated_view[:, i]
			result_index += 1
	
	# Compute the associated scores if necessary
	cdef double[:] result_scores_view
	if scores is not None:
		result_scores = np.empty(result_size)
		result_scores_view = result_scores
		result_index = 0
		for i in range(curve.shape[1]):
			if mask[i]:
				result_scores_view[result_index] = scores[i]
				result_index += 1
	
	free(dilated)
	free(mask)
	
	if scores is not None:
		return result, result_scores
	else:
		return result

def strip_angles(double[:, :] curve, double max_angle, double[:] scores=None):
	"""Cut out the points that make too sharp angles in the curve
	   - curve     : double[2, N] : Discrete curve to process
	   - max_angle : double       : Cut angles that are sharper than this (so, lower angle value), in radians
	   - scores    : double[N]    : Scores associated to each point of the curve, or None
	<--------------- double[2, M] : Curve with sharp angles deleted
	<--------------- double[M]    : ONLY IF `scores` is not None, scores associated to each point of the resulting curve, with the same points cut out
	"""
	cdef vector[Py_ssize_t] valid_indices
	cdef Py_ssize_t i
	cdef double left_norm, right_norm, angle
	cdef double[2] left_vector, right_vector
	for i in range(1, curve.shape[1]-1):
		# Compute the angle between the outgoing vectors
		left_vector[0] = curve[0, i-1] - curve[0, i]
		left_vector[1] = curve[1, i-1] - curve[1, i]
		right_vector[0] = curve[0, i+1] - curve[0, i]
		right_vector[1] = curve[1, i+1] - curve[1, i]
		angle = vector_angle(left_vector, right_vector)
		if angle > max_angle:
			valid_indices.push_back(i)
	
	# Copy the resulting point to a fixed numpy array
	# Don’t forget to include the initial and final points
	result = np.empty((2, valid_indices.size() + 2))
	cdef double[:, :] result_view = result
	result_view[:, 0] = curve[:, 0]
	result_view[:, valid_indices.size()+1] = curve[:, curve.shape[1]-1]
	cdef size_t result_index
	for result_index in range(valid_indices.size()):
		result_view[:, result_index+1] = curve[:, valid_indices[result_index]]
	
	# If necessary, also filter the scores
	cdef double[:] result_scores_view
	if scores is not None:
		result_scores = np.empty(valid_indices.size() + 2)
		result_scores_view = result_scores
		result_scores_view[0] = scores[0]
		result_scores_view[valid_indices.size()+1] = scores[scores.shape[0]-1]
		for result_index in range(valid_indices.size()):
			result_scores_view[result_index+1] = scores[valid_indices[result_index]]
		return result, result_scores
	else:
		return result

cpdef int savgol_window(int base_size, int array_size):
	"""Compute the best window size of the Savitzky-Golay filter for the given array size
	   - base_size  : int : Initial target window size
	   - array_size : int : Array size the filter has to be applied to
	<---------------- int : Optimal window size for the array (odd value, < array_size)"""
	if base_size < array_size:
		return base_size - (1 - base_size % 2)  # Make it odd
	else:  # Array size smaller than the window size : use the array size as window size
		return array_size - (1 - array_size % 2)


cdef void _savgol_coeffs_degree23(double[:] coeffs, int window_size):
	"""Compute the Savitzky-Golay filter coefficients for the given `window_size` and polynomial degree 2 or 3,
	   and put them in `coeffs`. There are `window_size` coefficients"""
	cdef Py_ssize_t i, i_first = (1 - window_size) // 2, i_last = (window_size - 1) // 2
	for i in range(i_first, i_last + 1):
		coeffs[i - i_first] = ((3 * window_size*window_size - 7 - 20 * i*i) / 4) / (window_size * (window_size**2 - 4) / 3)


# Cache for reused Savitzky-Golay coefficients
# TODO : Use a C++ map
cdef dict _savgol_coeffs_cache = {}

def savgol_filter(double[:, :] curve, int window_size, int savgol_degree):
	"""Apply a Savitzky-Golay filter on the given curve samples
	   - curve         : double[2, N] : Numerical curve to smooth
	   - window_size   : int          : Filter window size, must be odd and greater than `savgol_degree`
	   - savgol_degree : int          : Polynomial degree of the filter. Currently only support 2 and 3
	<------------------- double[2, N] : Filtered curve samples
	"""
	# Check the assumptions of this implementation
	assert savgol_degree in (2, 3)
	assert window_size % 2 == 1
	assert window_size <= curve.shape[1] and window_size > savgol_degree

	# Compute the convolution coefficients, or get them from the cache
	if (window_size, savgol_degree) in _savgol_coeffs_cache:
		coeffs = _savgol_coeffs_cache[(window_size, savgol_degree)]
	else:
		coeffs = np.empty(window_size)
		_savgol_coeffs_degree23(coeffs, window_size)
		_savgol_coeffs_cache[(window_size, savgol_degree)] = coeffs
	
	# Now apply a simple convolution to the curve
	# To avoid squishing the extreme points, we extrapolate the curve using its extreme vectors
	cdef Py_ssize_t row, i, x, npoints = curve.shape[1], nrows = curve.shape[0], i_first = (1 - window_size) // 2, conv_index
	cdef double conv_x
	cdef double[:] coeffs_view = coeffs
	result = np.zeros((curve.shape[0], curve.shape[1]))
	cdef double[:, :] result_view = result
	cdef double start_vector, end_vector, conv_value
	for row in range(nrows):
		start_vector = curve[row, 0] - curve[row, 1]
		end_vector = curve[row, npoints-1] - curve[row, npoints-2]
		for x in range(npoints):
			for i in range(window_size):
				# Index of the point to multiply this coefficient with
				conv_index = x + i + i_first

				# Index < 0 -> before the first sample. Extend with the first vector
				if conv_index < 0:
					conv_value = curve[row, 0] - start_vector*conv_index
				# Index >= npoints -> beyond the last sample. Extend with the last vector
				elif conv_index >= npoints:
					conv_value = curve[row, npoints-1] + end_vector*(conv_index - (npoints-1))
				else:
					conv_value = curve[row, conv_index]
				result_view[row, x] += conv_value * coeffs_view[i]
	return result

cpdef double mean_curvature(double[:, :] curve):
	"""Compute the mean curvature of the given curve, in rad/unit
	   - curve : double[2, N] : Curve to process
	<----------- Mean of the local curvatures in rad/length unit
	"""
	cdef Py_ssize_t i, valid_points = 0
	cdef double[2] prev_vector, next_vector
	cdef double curvature_sum = 0
	# Our local curvature is the angle between the incoming and outgoing vectors, divided by the mean local segment length
	for i in range(1, curve.shape[1]-1):
		prev_vector = [curve[0, i] - curve[0, i-1], curve[1, i] - curve[1, i-1]]
		next_vector = [curve[0, i+1] - curve[0, i], curve[1, i+1] - curve[1, i]]
		prev_vector_norm = sqrt(prev_vector[0]*prev_vector[0] + prev_vector[1]*prev_vector[1])
		next_vector_norm = sqrt(next_vector[0]*next_vector[0] + next_vector[1]*next_vector[1])

		curvature_sum += vector_angle(prev_vector, next_vector) / ((prev_vector_norm + next_vector_norm) / 2)
		valid_points += 1
	return curvature_sum / valid_points
	