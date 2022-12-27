# distutils: language=c++
# cython: boundscheck=False, wraparound=False

"""
Module with the functions that are really specific to this program,
and that couldn’t be left in Python
"""

import rospy
import numpy as np
from scipy.spatial import cKDTree
from libc.math cimport sqrt, exp, atan2, M_PI, INFINITY, acos, floor, ceil, cos, sin
from libc.stdlib cimport malloc, free
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from cython.operator cimport postincrement, dereference

import trajeometry
cimport trajeometry

cdef extern from "util.hpp":
	cdef vector[size_t] argsort[T](const vector[T])

cdef extern from "<algorithm>" namespace "std":
	T max[T](T a, T b)
	T min[T](T a, T b)

cdef double _acos_clip(double val):
	if val < -1:
		return acos(-1)
	elif val > 1:
		return acos(1)
	else:
		return acos(val)

def init_scores(double[:, :] line, double base_score, double score_range, double dampening):
	"""Initialize the reliability score of each point of the curve, given a score for the whole curve
	   - line       : ndarray[2, N] : Discrete curve
	   - base_score : float         : Score for the entire `line`
	<---------------- ndarray[N]    : Reliability scores associated to each point of the curve
	"""
	scores = np.ones(line.shape[1]) * base_score
	cdef double[:] scores_view = scores
	cdef double distance
	cdef Py_ssize_t i

	# Then the points beyond the reference index
	distance = 0
	for i in range(1, line.shape[1]):
		distance += sqrt((line[0, i-1] - line[0, i])**2 + (line[1, i-1] - line[1, i])**2)
		scores_view[i] *= max(0.0, 1 - exp((distance - score_range) / dampening))

	return scores

# TODO : Use the initial points to avoid weird angles in trajectory compilation
def line_parameters(list lines, double lane_width, double main_angle, double main_angle_distance, double angle_tolerance):
	"""Compute the parameters given to the fuzzy systems from the discrete curves
	   - lines               : list<ndarray[2, N]> : Input discrete curves, in metric coordinates in the local road frame
	   - lane_width          : double              : Expected lane width, in meters
	   - main_angle          : double              : Expected angle of the lane relative to the vehicle, in radians
	   - main_angle_distance : double              : Distance at which that angle is reached
	   - angle_tolerance     : double              : Tolerance in radians around the main_angle for the initial point of each curve
	<------------------------- ndarray[M]          : Distance of the initial point of each curve to the vehicle
	<------------------------- ndarray[M]          : Orthogonal distance of each initial point to the expected left lane marking
	<------------------------- ndarray[M]          : Orthogonal distance of each initial point to the expected right lane marking
	<------------------------- ndarray[M]          : Length of each curve from their initial point
	<------------------------- ndarray[M, M]       : Average distance between each pair of curves, in terms of expected lane widths
	<------------------------- ndarray[M, M]       : Average angle between each pair of curves, in radians
	"""
	cdef Py_ssize_t i, j, p, nlines = len(lines)
	
	# Allocate all those arrays
	forward_distance = np.empty(nlines)
	left_line_distance = np.empty(nlines)
	right_line_distance = np.empty(nlines)
	line_lengths = np.empty(nlines)
	parallel_distance = np.empty((nlines, nlines))
	parallel_angles = np.empty((nlines, nlines))

	cdef double[:] forward_distance_view = forward_distance
	cdef double[:] left_line_distance_view = left_line_distance
	cdef double[:] right_line_distance_view = right_line_distance
	cdef double[:] line_lengths_view = line_lengths
	cdef double[:, :] parallel_distance_view = parallel_distance
	cdef double[:, :] parallel_angles_view = parallel_angles
	cdef long[:] initial_indices = np.empty(nlines, dtype=int)

	cdef double[:, :] line_view
	cdef double vector_x, vector_y, vector_angle
	cdef Py_ssize_t initial_index

	# Precompute the expected markings
	# This is a bit complicated because assuming they are at `main_angle` straight ahead would completely disregard
	# the lateral offset and rotation of the vehicle relative to the lane

	# So what we do here, is fitting a circle arc from the current vehicle position,
	# such that it reaches `main_angle` relative to the vehicle when y = main_angle_distance
	# This gives us a lateral offset for the expected lane center, that gets at [offset, main_angle_distance] instead of [0, main_angle_distance]
	# Then we get base points for the left and right markings by orthogonally offsetting this main point by a half lane width on each side 
	# From that, the expected markings are defined by those main points and the expected vector extracted from the main angle
	# This complicates things a bit, but given our far, far away visibility window, it is unavoidable
	cdef double[2] expected_vector = [cos(main_angle), sin(main_angle)]
	cdef double[2] main_point = [0, main_angle_distance], left_main_point, right_main_point
	if main_angle < M_PI/2:  # Curving to the right
		main_point[0] = (main_angle_distance / cos(main_angle)) * (1 - sin(main_angle))
		left_main_point = [main_point[0] - expected_vector[1]*lane_width/2, main_point[1] + expected_vector[1]*lane_width/2]
		right_main_point = [main_point[0] + expected_vector[1]*lane_width/2, main_point[1] - expected_vector[1]*lane_width/2]
	elif main_angle > M_PI/2:  # Curving to the left
		main_point[0] = (main_angle_distance / cos(main_angle)) * (sin(main_angle) - 1)
		left_main_point = [main_point[0] + expected_vector[1]*lane_width/2, main_point[1] - expected_vector[1]*lane_width/2]
		right_main_point = [main_point[0] - expected_vector[1]*lane_width/2, main_point[1] + expected_vector[1]*lane_width/2]
	
	# Now, calculate the parameters for each curve
	for i in range(nlines):
		line_view = lines[i]

		# First, get the "initial point"
		# This is the first point at which the relative angle between the curve and the vehicle gets
		# reasonably close to the main angle, to eliminate completely irrelevant curves right avay
		# and disregard weird starts for some curves
		initial_index = -1
		for p in range(line_view.shape[1] - 1):
			vector_x = line_view[0, p+1] - line_view[0, p]
			vector_y = line_view[1, p+1] - line_view[1, p]

			vector_angle = atan2(vector_y, vector_x)
			if main_angle - angle_tolerance < vector_angle and vector_angle < main_angle + angle_tolerance:
				initial_index = p
				break
				
		# The curve is completely out of line : make it definitely invalid for the fuzzy system
		if initial_index < 0:
			initial_indices[i] = -1
			forward_distance_view[i] = INFINITY
			left_line_distance_view[i] = INFINITY
			right_line_distance_view[i] = INFINITY
			line_lengths_view[i] = 0
			continue
		
		# Now compute the parameters
		initial_indices[i] = initial_index
		forward_distance_view[i] = line_view[1, initial_index]
		
		# Orthogonal distance to a line defined by an angle and a reference point
		left_line_distance_view[i] = abs(expected_vector[0]*(line_view[1, initial_index] - left_main_point[1]) - expected_vector[1]*(line_view[0, initial_index] - left_main_point[0])) / lane_width
		right_line_distance_view[i] = abs(expected_vector[0]*(line_view[1, initial_index] - right_main_point[1]) - expected_vector[1]*(line_view[0, initial_index] - right_main_point[0])) / lane_width
		line_lengths_view[i] = trajeometry.line_length(line_view[:, initial_index:])

	# Now that we have the individual parameters, we need to compute the pairwise parameters
	# (parallel distance and angles)
	cdef double[:, :] longest_line, shortest_line
	cdef trajeometry._ProjectionIndex projection
	cdef double vector1_x, vector1_y, vector2_x, vector2_y, anglediff, paralleldiff
	cdef int valid_points
	for i in range(nlines):
		parallel_angles_view[i, i] = INFINITY
		parallel_distance_view[i, i] = INFINITY

		for j in range(i+1, nlines):
			# The lines are not in front of each other -> invalid pair
			if initial_indices[i] < 0 or initial_indices[j] < 0:
				parallel_angles_view[i, j] = parallel_angles_view[j, i] = INFINITY
				parallel_distance_view[i, j] = parallel_distance_view[j, i] = INFINITY
				continue
			
			if line_lengths_view[i] > line_lengths_view[j]:
				longest_line = lines[i]
				shortest_line = lines[j]
			else:
				longest_line = lines[j]
				shortest_line = lines[i]
			
			# Now compute the average orthogonal distance and the average angle difference
			paralleldiff = 0
			anglediff = 0
			valid_points = 0
			for p in range(longest_line.shape[1] - 1):
				projection = trajeometry._project_on_curve_index(longest_line[0, p], longest_line[1, p], shortest_line, False, 0)
				if projection.index < 0:
					continue
				
				vector1_x = longest_line[0, p+1] - longest_line[0, p]
				vector1_y = longest_line[1, p+1] - longest_line[1, p]
				vector2_x = shortest_line[0, projection.index+1] - shortest_line[0, projection.index]
				vector2_y = shortest_line[1, projection.index+1] - shortest_line[1, projection.index]
				
				anglediff += abs(_acos_clip((vector1_x*vector2_x + vector1_y*vector2_y) / (sqrt(vector1_x*vector1_x + vector1_y*vector1_y) * sqrt(vector2_x*vector2_x + vector2_y*vector2_y))))
				paralleldiff += abs(sqrt((longest_line[0, p] - shortest_line[0, projection.index])**2 + (longest_line[1, p] - shortest_line[1, projection.index])**2) - lane_width) / lane_width
				valid_points += 1
			
			if valid_points == 0:
				parallel_angles_view[i, j] = parallel_angles_view[j, i] = INFINITY
				parallel_distance_view[i, j] = parallel_distance_view[j, i] = INFINITY
			else:
				parallel_angles_view[i, j] = parallel_angles_view[j, i] = anglediff / valid_points
				parallel_distance_view[i, j] = parallel_distance_view[j, i] = paralleldiff / valid_points
	
	return forward_distance, left_line_distance, right_line_distance, line_lengths, parallel_distance, parallel_angles


def compile_line(list lines, list scores, double min_score, (double, double) start_point, double trajectory_step):
	"""Compile a single curve and its reliability scores from a group of discrete curves
		- lines           : list<ndarray[2, N]> : Curves to compile
		- scores          : list<ndarray[N]>    : Scores for each point of each curve
		- min_score       : float               : Points below this score will not be considered
		- start_point     : ndarray[2]          : Initial point of the compiled curve
		- trajectory_step : double              : Interval at which the points are taken
	<---------------------- ndarray[2, M]       : Compiled curve
	<---------------------- ndarray[M]          : Scores for each point of the compiled curve
	"""
	cdef size_t i, nlines = len(lines)
	
	if nlines == 0:  # No line to compile, so compile nothing
		return np.zeros((2, 0)), np.zeros(0)

	cdef vector[(double, double)] result_points
	cdef vector[double] result_scores

	cdef double[2] last_centroid = [start_point[0], start_point[1]]
	cdef double[2] centroid, result_vector, previous_vector
	cdef long* point_indices = <long*>malloc(nlines * sizeof(long))
	for i in range(nlines):
		point_indices[i] = 0

	# Define a max length to be able to break the loop if the function converges undetected,
	# and a max Y coordinate to break the loop if there are definitely no interesting points at all
	cdef double length, max_y = 0, max_expected_length = 0, result_length = 0
	for i in range(nlines):
		length = 2 * trajeometry.line_length(lines[i]) / trajectory_step
		if length > max_expected_length:
			max_expected_length = length
		if lines[i][1, lines[i].shape[1]-1] > max_y:
			max_y = lines[i][1, lines[i].shape[1]-1]

	cdef trajeometry._NextPoint next_point
	cdef double score, centroid_numerator_x, centroid_numerator_y, centroid_denominator
	cdef vector[(double, double)] next_points
	cdef vector[double] next_scores
	cdef vector[double] next_distances
	cdef vector[size_t] next_argsort
	cdef double quartile1_index, quartile3_index, quartile1, quartile3, iqr_filter
	cdef double centroid_score
	cdef int valid_points
	while True:
		next_points.clear()
		next_scores.clear()

		centroid_numerator_x = 0
		centroid_numerator_y = 0
		centroid_denominator = 0
		for i in range(nlines):
			# Project the last centroid onto each line and take the point that is `trajectory-step` further along it
			# We also compute a first estimate of the weighted centroid for the IQR filter
			next_point, score = trajeometry._next_point_score(last_centroid, lines[i], scores[i], trajectory_step, extend=True, min_index=point_indices[i])
			if next_point.index >= 0 and next_point.index >= point_indices[i] and score > min_score:
				point_indices[i] = next_point.index
				next_points.push_back((next_point.x, next_point.y))
				next_scores.push_back(score)
				centroid_numerator_x += score * next_point.x
				centroid_numerator_y += score * next_point.y
				centroid_denominator += score
		
		# No points interesting enough
		# If there are no points in the result list yet, the closest points might just be too far extrapolated, 
		# so skip to the next step to see if it is better, and cut if it has gone the whole way without finding anything
		# If there are already result points, stop here as it was most likely the last interesting position
		if next_points.size() == 0:
			if result_points.size() == 0 and last_centroid[1] < max_y:
				last_centroid[1] += trajectory_step
				continue
			else:
				break
		
		# One point, the centroid is easy to compute
		elif next_points.size() == 1:
			centroid[0] = next_points[0][0]
			centroid[1] = next_points[0][1]
			centroid_score = next_scores[0]
		else:
			# Filter the valid points by performing an interquartile range filtering about the distance to the estimated centroid
			next_distances.clear()
			centroid[0] = centroid_numerator_x / centroid_denominator
			centroid[1] = centroid_numerator_y / centroid_denominator

			for i in range(next_points.size()):
				next_distances.push_back(sqrt((next_points[i][0] - centroid[0])**2 + (next_points[i][1] - centroid[1])**2))

			# Extract the quartiles and calculate the interquartile range filter
			next_argsort = argsort(next_distances)
			if (next_argsort.size() - 1) % 4 == 0:
				quartile1 = next_distances[next_argsort[(next_argsort.size() - 1) // 4]]
				quartile3 = next_distances[next_argsort[3 * (next_argsort.size() - 1) // 4]]
			else:
				quartile1_index = (next_argsort.size() - 1) / 4.0
				quartile3_index = 3.0 * (next_argsort.size() - 1) / 4.0
				quartile1 = (ceil(quartile1_index) - quartile1_index) * next_distances[next_argsort[<Py_ssize_t>floor(quartile1_index)]] + (quartile1_index - floor(quartile1_index)) * next_distances[next_argsort[<Py_ssize_t>ceil(quartile1_index)]]
				quartile3 = (ceil(quartile3_index) - quartile3_index) * next_distances[next_argsort[<Py_ssize_t>floor(quartile3_index)]] + (quartile3_index - floor(quartile3_index)) * next_distances[next_argsort[<Py_ssize_t>ceil(quartile3_index)]]
			iqr_filter = quartile1 + 1.5 * (quartile3 - quartile1)

			# Now eliminate all points further than that IQR filter to the estimated centroid,
			# and update the centroid to account for it
			centroid_numerator_x = 0
			centroid_numerator_y = 0
			centroid_denominator = 0
			centroid_score = 0
			valid_points = 0
			for i in range(next_distances.size()):
				if next_distances[i] < iqr_filter:
					centroid_numerator_x += next_scores[i]*next_points[i][0]
					centroid_numerator_y += next_scores[i]*next_points[i][1]
					centroid_denominator += next_scores[i]
					centroid_score += (1 - next_scores[i])**2
					valid_points += 1
			
			if valid_points == 0:
				break
		
			# Finally, compute the final weighted centroid and its cumulated score
			centroid[0] = centroid_numerator_x / centroid_denominator
			centroid[1] = centroid_numerator_y / centroid_denominator
			centroid_score = 1 - sqrt(centroid_score) / valid_points


		# When some of the initial curves are not well defined, like going in opposite directions as some point,
		# it might make the centroids converge and make this function go into an infinite loop
		# Here are some safeguards against this

		# Prevent direct convergence by cutting when the steps become too small
		if sqrt((centroid[0] - last_centroid[0])**2 + (centroid[1] - last_centroid[1])**2) < trajectory_step / 20:
			rospy.logwarn("Curve compilation convergence (direct)")
			break

		# Prevent oscillating convergence by cutting when the angle between the two last vectors is higher than 2π/3,
		# so when the centroid goes back on its tracks
		if result_points.size() >= 2:
			result_vector[0] = centroid[0] - result_points.back()[0]
			result_vector[1] = centroid[1] - result_points.back()[1]
			previous_vector[0] = result_points.back()[0] - result_points[result_points.size() - 2][0]
			previous_vector[1] = result_points.back()[1] - result_points[result_points.size() - 2][1]
			if _acos_clip((result_vector[0]*previous_vector[0] + result_vector[1]*previous_vector[1]) / (sqrt(result_vector[0]*result_vector[0] + result_vector[1]*result_vector[1]) * sqrt(previous_vector[0]*previous_vector[0] + previous_vector[1]*previous_vector[1]))) > 2*M_PI/3:
				rospy.logwarn("Curve compilation convergence (oscillating)")
				break

		if result_points.size() > 0:
			result_length += sqrt((centroid[0] - result_points.back()[0])**2 + (centroid[1] - result_points.back()[1])**2)

		# If all else have failed, cut when we attain an inordinate amount of points
		if result_length > max_expected_length:
			rospy.logwarn("Curve compilation convergence (unexplained)")
			break

		result_points.push_back((centroid[0], centroid[1]))
		result_scores.push_back(centroid_score)

		last_centroid[0] = centroid[0]
		last_centroid[1] = centroid[1]
	
	# We have our results in a vector, put them in a fixed numpy array
	result_line_array = np.empty((2, result_points.size()))
	result_score_array = np.empty(result_scores.size())
	cdef double[:, :] result_line_view = result_line_array
	cdef double[:] result_score_view = result_score_array
	for i in range(result_points.size()):
		result_line_view[0, i] = result_points[i][0]
		result_line_view[1, i] = result_points[i][1]
		result_score_view[i] = result_scores[i]
	
	free(point_indices)
	return result_line_array, result_score_array


cdef bint _pca_2x2(double[:, :] points, double[2] eigenvalues, double[2][2] eigenvectors) noexcept:
	"""Fast 2D PCA transform matrix computation (at least, a lot faster than numpy)
	   The algorithms are mostly pulled from here : https://en.wikipedia.org/wiki/Eigenvalue_algorithm#2.C3.972_matrices
	   - points       : double[2, N] : Points of which to eigendecompose
	   - eigenvalues  : double[2]    : Output for the eigenvalues, sorted by descending magnitude
	   - eigenvectors : double[2][2] : Output for the eigenvectors, as column vectors corresponding to the respective `eigenvalues`
	<------------------ bool         : True if the eigendecomposition was successful, False otherwise
	                                   If False, the content of `eigenvalues` and `eigenvectors` is valid,
									   but with some mathematical problem that make them unusable for PCA decomposition (null eigenvectors, …)"""
	cdef double[2][2] covariance = [[0, 0], [0, 0]]
	cdef double[2] mean = [0, 0]
	cdef double cov_value
	cdef Py_ssize_t i
	
	# Compute the covariance matrix, (P - μ) × (P - μ)ᵀ
	for i in range(points.shape[1]):
		mean[0] += points[0, i]
		mean[1] += points[1, i]
	mean[0] /= points.shape[1]
	mean[1] /= points.shape[1]
	
	for i in range(points.shape[1]):
		covariance[0][0] += (points[0, i] - mean[0])**2
		covariance[1][1] += (points[1, i] - mean[1])**2
		covariance[0][1] += (points[0, i] - mean[0]) * (points[1, i] - mean[1])
	covariance[0][0] /= points.shape[1] - 1
	covariance[1][1] /= points.shape[1] - 1
	covariance[0][1] /= points.shape[1] - 1
	
	# Then its eigenvalues with the trace method :
	# The first is from the annihilating polynomial, and as the trace of a diagonalisable matrix is the sum of its
	# eigenvalues, the second eigenvalue is just trace - λ₁
	cdef double cov_trace = covariance[0][0] + covariance[1][1]
	cdef double cov_determinant = covariance[0][0]*covariance[1][1] - covariance[0][1]*covariance[0][1]
	eigenvalues[0] = (cov_trace + sqrt(cov_trace*cov_trace - 4*cov_determinant)) / 2
	eigenvalues[1] = cov_trace - eigenvalues[0]
	
	if eigenvalues[1] > eigenvalues[0]:
		eigenvalues[0], eigenvalues[1] = eigenvalues[1], eigenvalues[0]
	
	# Now get the eigenvectors simply from the covariance matrix, using C - λ₁I₂ and C - λ₂I₂
	eigenvectors[0][0] = covariance[0][0] - eigenvalues[1]
	eigenvectors[0][1] = covariance[0][1]
	eigenvectors[1][0] = covariance[0][1]
	eigenvectors[1][1] = covariance[1][1] - eigenvalues[0]
	
	# Normalize the eigenvectors
	cdef double ev1_norm = sqrt(eigenvectors[0][0]*eigenvectors[0][0] + eigenvectors[1][0]*eigenvectors[1][0])
	cdef double ev2_norm = sqrt(eigenvectors[0][1]*eigenvectors[0][1] + eigenvectors[1][1]*eigenvectors[1][1])
	if ev1_norm == 0 or ev2_norm == 0:
		return False
	
	eigenvectors[0][0] /= ev1_norm
	eigenvectors[1][0] /= ev1_norm
	eigenvectors[0][1] /= ev2_norm
	eigenvectors[1][1] /= ev2_norm
	return True

cdef void _inverse_2x2(double[2][2] matrix, double[2][2] inverse) noexcept nogil:
	"""Fast inverse for 2×2 matrices (at least, a lot faster than numpy)
	   - matrix  : double[2][2] : 2×2 matrix to inverse
	   - inverse : double[2][2] : Output for the inverse matrix
	"""
	cdef double determinant = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
	inverse[0][0] = matrix[1][1] / determinant
	inverse[0][1] = -matrix[0][1] / determinant
	inverse[1][0] = -matrix[1][0] / determinant
	inverse[1][1] = matrix[0][0] / determinant

# TODO : Own implementation of the k-d tree, bruteforce method as there should not be too much point ?
cdef void _cluster_DBSCAN(double[:, :] data, long[:] labels, double epsilon, long min_samples):
	"""Perform a DBSCAN clustering of the given data points, a lot faster than sklearn
	   - data : double[N, k] : Data points as LINE VECTORS of `k` features
	   - labels : long[N] : Labels given to each corresponding data point, with respect to its cluster
	                        -1 indicates a point that is not in any cluster
                            Contrary to sklearn, those labels are not sequential ’cuz it’s easier
	   - epsilon : double : Max distance between two points to consider them neighbors
	   - min_samples : double : Min number of points in a cluster to consider it a cluster and not noise
	"""
	# Make the nearest-neighbor query for each point, currently with the scipy k-d tree that goes back and forth multiple times
	# between C and Python 
	tree = cKDTree(data)
	cdef list neighbors = tree.query_ball_tree(tree, epsilon)

	# Initialize each point in its own cluster
	cdef Py_ssize_t i, j, index
	for i in range(data.shape[0]):
		labels[i] = i

	# Then join clusters when two points from different clusters are in the neighborhood of each other 
	cdef list neighborhood
	cdef long neighborhood_size, result_label, replaced_label
	for i in range(data.shape[0]):
		neighborhood = neighbors[i]
		neighborhood_size = len(neighborhood)
		if neighborhood_size <= 1:  # No neighbors
			continue

		# Now, for each neighbor, replace the whole neighbor’s cluster by the current point’s cluster, to merge the clusters
		result_label = labels[<Py_ssize_t>neighborhood[0]]
		for j in range(neighborhood_size):
			index = neighborhood[j]
			replaced_label = labels[index]
			if replaced_label == result_label:  # Already in the same cluster
				continue
			for j in range(data.shape[0]):
				if labels[j] == replaced_label:
					labels[j] = result_label
	
	# Count the points in each cluster
	cdef long* label_counts = <long*>malloc(data.shape[0] * sizeof(long))
	for i in range(data.shape[0]):
		label_counts[i] = 0
	
	for i in range(data.shape[0]):
		label_counts[labels[i]] += 1
	
	# And eliminate the clusters that have too few points, making them into noise (label -1)
	cdef long current_label = 0
	for i in range(data.shape[0]):
		if label_counts[i] < min_samples:
			for j in range(data.shape[0]):
				if labels[j] == i:
					labels[j] = -1
	free(label_counts)

cdef vector[Py_ssize_t] _compact_array(double[:, :] array, bint[:] mask) noexcept nogil:
	"""Push the relevant element of the array contiguously at the front, based on a mask
	   The elements after the last relevant item are undefined
	   - array : double[:, :] : Array of vectors to compact, regardless of the vector dimension
	   - mask  : bool[:]      : Mask, with 1 at the indices of items to keep, and 0 for those to remove
	<----------- vector<Py_ssize_t> : Index mapping from the compacted array to its former state
	                                  For instance, index `n` in the compacted array was formerly at index `return_value[n]`"""
	cdef vector[Py_ssize_t] indices
	cdef Py_ssize_t read_index, write_index = 0
	for read_index in range(array.shape[0]):
		if mask[read_index]:
			indices.push_back(read_index)
			if write_index != read_index:
				array[write_index] = array[read_index]
			write_index += 1
	return indices


DEF EV_SIDE_BINS = 7
DEF EV2_BINS = 15

# TODO : Use parameters
# TODO : Detect dotted lines
# TODO : Detect other rectangular lane markings
def find_markings((long, long) be_shape, list branches, double scale_factor, double crosswalk_width, double size_tolerance):
	"""Extract road markings when possible
		- be_binary       : ndarray[y, x]       : Preprocessed bird-eye view
		- branches        : list<ndarray[2, N]> : List of discrete curves detected in the image
		- scale_factor    : double              : Scale factor from pixel to metric in the bird-eye view
		- crosswalk_width : double              : Expected width of a crosswalk
		- size_tolerance  : double              : Tolerance percentage around expected sizes
	<---------------------- dict<str: …>        : Detected markings sorted by marking type :
												  - `crosswalks` : Crosswalks, elements are lists of ndarray[2, 4] rectangle corners
	<---------------------- list<ndarray[2, N]> : `branches` with consumed branches removed
	"""
	cdef Py_ssize_t i, j, nbranches = len(branches)
	# Nothing to detect from
	if nbranches == 0:
		return {}, branches

	# Mask to keep track of which branches have been used for markings or not. True is when it has *not* been used
	cdef bint* available = <bint*>malloc(nbranches * sizeof(bint))
	cdef bint* is_rectangle = <bint*>malloc(nbranches * sizeof(bint))
	cdef bint* possible_crosswalk_ptr = <bint*>malloc(nbranches * sizeof(bint))
	cdef bint* possible_dotline_ptr = <bint*>malloc(nbranches * sizeof(bint))
	
	cdef bint[::1] possible_crosswalk = <bint[:nbranches:1]> possible_crosswalk_ptr
	cdef bint[::1] possible_dotline = <bint[:nbranches:1]> possible_dotline_ptr

	for i in range(nbranches):
		available[i] = True
		is_rectangle[i] = False
		possible_crosswalk[i] = False
		possible_dotline[i] = False

	# Initialize the heaps of variables necessary for all this
	cdef double* crosswalk_data_ptr = <double*>malloc(4*nbranches * sizeof(double))
	cdef double* dotline_data_ptr = <double*>malloc(4*nbranches * sizeof(double))
	cdef double* rectangle_corners_ptr = <double*>malloc(2*4*nbranches * sizeof(double))
	cdef double* rectangle_errors_ptr = <double*>malloc(nbranches * sizeof(double))

	cdef double[:, ::1] crosswalk_data = <double[:nbranches, :4:1]> crosswalk_data_ptr
	cdef double[:, ::1] dotline_data = <double[:nbranches, :4:1]> dotline_data_ptr
	cdef double[:, :, ::1] rectangle_corners = <double[:nbranches, :2, :4:1]> rectangle_corners_ptr
	cdef double[::1] rectangle_errors = <double[:nbranches:1]> rectangle_errors_ptr

	cdef double[:, :] branch, resampled
	cdef double[2] eigenvalues = [0, 0]
	cdef double[2][2] eigenvectors = [[0, 0], [0, 0]]
	cdef double[2][2] inverse_eigenvectors = [[0, 0], [0, 0]]

	cdef double main_angle, ev1_min, ev1_max, ev2_min, ev2_max
	cdef Py_ssize_t npoints
	cdef Py_ssize_t pca_alloc_size = -1
	cdef double* pca_ptr = NULL
	cdef double[:, ::1] pca

	cdef Py_ssize_t ev1_nbins, ev2_nbins = EV2_BINS, ev1_bin_index, ev2_bin_index
	cdef long[EV_SIDE_BINS] ev1_bins_left
	cdef long[EV_SIDE_BINS] ev1_bins_right
	cdef long[EV_SIDE_BINS] ev2_bins_top
	cdef long[EV_SIDE_BINS] ev2_bins_bottom

	cdef long max_bin_left, max_bin_right, max_bin_top, max_bin_bottom
	cdef double ev1_left, ev1_right, ev2_top, ev2_bottom, error
	cdef double direct_projection, cross_projection
	cdef double[2][4] pca_corners
	cdef double[2] centroid

	for i in range(nbranches):
		branch = branches[i]
		resampled = trajeometry.resample_curve(branch, 5)
		npoints = resampled.shape[1]
		if npoints < 3:
			continue

		# Perform the eigendecomposition
		if not _pca_2x2(resampled, eigenvalues, eigenvectors):
			continue
		
		if eigenvalues[0] == 0 or eigenvalues[1] == 0:
			continue

		# We need the angle of the principal component for later
		main_angle = atan2(eigenvectors[1][0], eigenvectors[0][0])
		
		# Now, we are going to transform all points to PCA space
		# So we need a buffer. If the last one is large enough, reuse it,
		# otherwise ditch it and allocate a larger one
		if npoints > pca_alloc_size:
			if pca_ptr != NULL:
				free(pca_ptr)
			pca_ptr = <double*> malloc(2 * npoints * sizeof(double))
			pca = <double[:2, :npoints:1]> pca_ptr
			pca_alloc_size = npoints
		
		# Transform all points into PCA space and find the minimum and maximum along both PCA axes by the way
		ev1_min = INFINITY
		ev1_max = -INFINITY
		ev2_min = INFINITY
		ev2_max = -INFINITY
		for j in range(npoints):
			pca[0, j] = eigenvectors[0][0] * resampled[0, j] + eigenvectors[0][1] * resampled[1, j]
			pca[1, j] = eigenvectors[1][0] * resampled[0, j] + eigenvectors[1][1] * resampled[1, j]
			if pca[0, j] > ev1_max:
				ev1_max = pca[0, j]
			if pca[0, j] < ev1_min:
				ev1_min = pca[0, j]
			if pca[1, j] > ev2_max:
				ev2_max = pca[1, j]
			if pca[1, j] < ev2_min:
				ev2_min = pca[1, j]
		
		# Null width or height : nothing interesting here
		if ev1_min == ev1_max or ev2_min == ev2_max:
			continue
		
		# Now, we take the vertical and horizontal histograms to find the most likely rectangle edges
		# This is easy, as in PCA space the longest edge is perfectly horizontal and the shortest is vertical
		# We take a fixed amount of bins for the secondary component,
		# but scale the number of bins of the principal component to make approximate squares 
		ev1_nbins = <Py_ssize_t>(ev2_nbins * eigenvalues[0] / eigenvalues[1])
		
		# Actually, we need only the first and last few bins at each side of each axis, so only store and initialize those
		for j in range(EV_SIDE_BINS):
			ev1_bins_left[j] = 0
			ev1_bins_right[j] = 0
			ev2_bins_top[j] = 0
			ev2_bins_bottom[j] = 0
		
		# Compute the relevant parts of the histogram
		for j in range(npoints):
			ev1_bin_index = <Py_ssize_t>(ev1_nbins * (pca[0, j] - ev1_min) / (ev1_max - ev1_min))
			ev2_bin_index = <Py_ssize_t>(ev2_nbins * (pca[1, j] - ev2_min) / (ev2_max - ev2_min))
			# ev1_max and ev2_max will report at exactly `ev1/2_nbins`, put them back in the actual last bin
			if ev1_bin_index >= ev1_nbins:
				ev1_bin_index = ev1_nbins - 1
			if ev2_bin_index >= ev2_nbins:
				ev2_bin_index = ev2_nbins - 1
			
			# Add the point to the relevant bins if it belongs to them
			if ev1_bin_index < EV_SIDE_BINS:
				ev1_bins_left[ev1_bin_index] += 1
			if ev1_bin_index >= ev1_nbins - EV_SIDE_BINS:
				ev1_bins_right[ev1_bin_index - (ev1_nbins - EV_SIDE_BINS)] += 1
			if ev2_bin_index < EV_SIDE_BINS:
				ev2_bins_top[ev2_bin_index] += 1
			if ev2_bin_index >= ev2_nbins - EV_SIDE_BINS:
				ev2_bins_bottom[ev2_bin_index - (ev2_nbins - EV_SIDE_BINS)] += 1
		
		# Fit the rectangle edges to the bin with highest number of points on each side of each axis
		max_bin_left = max_bin_right = -1
		max_bin_top = max_bin_bottom = -1
		ev1_left = ev1_right = -INFINITY
		ev2_top = ev2_bottom = -INFINITY

		for j in range(EV_SIDE_BINS):
			if ev1_bins_left[j] > max_bin_left:
				max_bin_left = ev1_bins_left[j]
				ev1_left = (j + 0.5) * (ev1_max - ev1_min) / ev1_nbins + ev1_min

			if ev1_bins_right[j] > max_bin_right:
				max_bin_right = ev1_bins_right[j]
				ev1_right = ((j + (ev1_nbins - EV_SIDE_BINS)) + 0.5) * (ev1_max - ev1_min) / ev1_nbins + ev1_min
			
			if ev2_bins_top[j] > max_bin_top:
				max_bin_top = ev2_bins_top[j]
				ev2_top = (j + 0.5) * (ev2_max - ev2_min) / ev2_nbins + ev2_min
			
			if ev2_bins_bottom[j] > max_bin_bottom:
				max_bin_bottom = ev2_bins_bottom[j]
				ev2_bottom = ((j + (ev2_nbins - EV_SIDE_BINS)) + 0.5) * (ev2_max - ev2_min) / ev2_nbins + ev2_min
		
		width = ev1_right - ev1_left
		height = ev2_bottom - ev2_top
		
		# Now compute the sum of squared distances of the fitted rectangle to the actual curve, still in PCA space
		# The distance of a curve point to the rectangle is its distance to the closest edge of the rectangle
		# And as those edges are just vertical and horizontal lines in PCA space, all this is easy to compute
		error = 0
		for j in range(npoints):
			error += min(min((pca[0, j] - ev1_left)**2, (pca[0, j] - ev1_right)**2),
			             min((pca[1, j] - ev2_top)**2,  (pca[1, j] - ev2_bottom)**2))
		error /= width * height  # Normalize by the rectangle area, otherwise it’s not fair to larger rectangles
		rectangle_errors[i] = error

		width *= scale_factor
		height *= scale_factor

		# Only keep relatively good rectangles that are at least 10×10cm
		if error < 1.6 and width > 0.1 and height > 0.1:
			is_rectangle[i] = True

			# Calculate the corners in PCA space, as the rectangle is orthogonal to the base it’s way easier here
			pca_corners[0][0] = pca_corners[0][3] = ev1_left
			pca_corners[0][1] = pca_corners[0][2] = ev1_right
			pca_corners[1][0] = pca_corners[1][1] = ev2_top
			pca_corners[1][2] = pca_corners[1][3] = ev2_bottom

			# Get the corners back in image space and compute the rectangle’s centroid
			_inverse_2x2(eigenvectors, inverse_eigenvectors)
			centroid[0] = centroid[1] = 0
			for j in range(4):
				rectangle_corners[i, 0, j] = pca_corners[0][j] * inverse_eigenvectors[0][0] + pca_corners[1][j] * inverse_eigenvectors[0][1]
				rectangle_corners[i, 1, j] = pca_corners[0][j] * inverse_eigenvectors[1][0] + pca_corners[1][j] * inverse_eigenvectors[1][1]
				centroid[0] += rectangle_corners[i, 0, j]
				centroid[1] += rectangle_corners[i, 1, j]
			centroid[0] /= 4
			centroid[1] /= 4

			# Now, to detect crosswalks and dotted lines, we project the center of the rectangle onto its eigenvectors
			# The direct projection is along the principal component, as crosswalk rectangles have their largest dimension parallel,
			# their centers project approximately at the same place on their principal component
			# Conversely, dotted lines have their secondary components parallel, so their centers project approximately at the same place on their secondary component
			direct_projection = (eigenvectors[0][0]*centroid[0] + eigenvectors[1][0]*centroid[1]) / (eigenvectors[0][0]*eigenvectors[0][0] + eigenvectors[1][0]*eigenvectors[1][0])
			#cross_projection = (eigenvectors[0][1]*centroid[0] + eigenvectors[1][1]*centroid[1]) / (eigenvectors[0][1]*eigenvectors[0][1] + eigenvectors[1][1]*eigenvectors[1][1])

			if crosswalk_width * (1 - size_tolerance) < height and height < crosswalk_width * (1 + size_tolerance):
				possible_crosswalk[i] = True
				crosswalk_data[i, 0] = width / (be_shape[1]*scale_factor)
				crosswalk_data[i, 1] = (height - crosswalk_width) / crosswalk_width
				crosswalk_data[i, 2] = main_angle / M_PI
				crosswalk_data[i, 3] = direct_projection / be_shape[0]
			#possible_dotline[i] = True
			#dotline_data[i, 0] = width / (be_shape[1]*scale_factor)
			#dotline_data[i, 1] = height / (be_shape[1]*scale_factor)
			#dotline_data[i, 2] = main_angle / M_PI
			#dotline_data[i, 3] = direct_projection / be_shape[0]
	
	#cdef vector[Py_ssize_t] dotline_indices = _compact_array(dotline_data, possible_dotline)
	#cdef Py_ssize_t n_possible_dotline = dotline_indices.size()
	
	cdef list crosswalks = []
	cdef vector[Py_ssize_t] crosswalk_indices = _compact_array(crosswalk_data, possible_crosswalk)
	cdef Py_ssize_t n_possible_crosswalk = crosswalk_indices.size()

	cdef long* crosswalk_labels_ptr = NULL
	cdef long[:] crosswalk_labels
	cdef long cluster_size
	cdef double[:, :, :] cluster_view
	cdef map[long, vector[long]] clusters
	cdef map[long, vector[long]].iterator cluster_it

	if n_possible_crosswalk > 0:
		# Crosswalk detection
		# Crosswalks are rows of rectangle, with approximately the same shape and angle
		# So cluster the possible crosswalk elements with DBSCAN and see what comes out
		crosswalk_labels_ptr = <long*>malloc(n_possible_crosswalk * sizeof(long))
		crosswalk_labels = <long[:n_possible_crosswalk:1]>crosswalk_labels_ptr
		_cluster_DBSCAN(crosswalk_data[:n_possible_crosswalk], crosswalk_labels, 0.1, 3)

		# Group the indices by cluster
		for i in range(n_possible_crosswalk):
			if crosswalk_labels[i] >= 0:
				clusters[crosswalk_labels[i]].push_back(i)
		
		# And add the rectangle corners as a list of numpy arrays
		cluster_it = clusters.begin()
		while cluster_it != clusters.end():
			cluster_size = dereference(cluster_it).second.size()
			cluster = np.empty((cluster_size, 2, 4))
			cluster_view = cluster
			for j in range(cluster_size):
				index = crosswalk_indices[dereference(cluster_it).second[j]]
				cluster_view[j] = rectangle_corners[index, :, :]
				available[index] = False
			crosswalks.append(cluster)
			postincrement(cluster_it)
		free(crosswalk_labels_ptr)

	cdef dict markings = {"crosswalks": crosswalks}
	cdef list remainder = [branch for i, branch in enumerate(branches) if available[i]]  # Unused branches

	# Free our heaps of buffers
	free(available)
	free(is_rectangle)
	free(possible_crosswalk_ptr)
	free(possible_dotline_ptr)
	free(crosswalk_data_ptr)
	free(dotline_data_ptr)
	free(rectangle_corners_ptr)
	free(rectangle_errors_ptr)
	if pca_ptr != NULL:
		free(pca_ptr)		

	return markings, remainder