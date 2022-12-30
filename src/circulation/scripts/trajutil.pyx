# distutils: language=c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False

from scipy.spatial import cKDTree

cimport cython
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector


cdef bint pca_2x2(double[:, :] points, double[2] eigenvalues, double[2][2] eigenvectors) noexcept:
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
	
	# Compute the covariance matrix, 1/n (P - μ) × (P - μ)ᵀ
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

cdef void inverse_2x2(double[2][2] matrix, double[2][2] inverse) noexcept nogil:
	"""Fast inverse for 2×2 matrices (at least, a lot faster than numpy)
	   - matrix  : double[2][2] : 2×2 matrix to inverse
	   - inverse : double[2][2] : Output for the inverse matrix
	"""
	cdef double determinant = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
	inverse[0][0] = matrix[1][1] / determinant
	inverse[0][1] = -matrix[0][1] / determinant
	inverse[1][0] = -matrix[1][0] / determinant
	inverse[1][1] = matrix[0][0] / determinant

# TODO : Own implementation of the k-d tree, bruteforce method as there should not be too many points ?
cdef void cluster_DBSCAN(double[:, :] data, long[:] labels, double epsilon, long min_samples):
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
	cdef set pairs = tree.query_pairs(epsilon)

	# Initialize each point in its own cluster
	cdef Py_ssize_t i, j, index
	for i in range(data.shape[0]):
		labels[i] = i

	# Then join clusters when two points from different clusters are in the neighborhood of each other 
	cdef long neighbor1, neighbor2
	cdef long result_label, replaced_label
	for neighbor1, neighbor2 in pairs:
		# Replace the whole neighbor’s cluster by the current point’s cluster, to merge the clusters
		result_label = labels[neighbor1]
		replaced_label = labels[neighbor2]
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

cdef vector[Py_ssize_t] compact_array(double[:, :] array, bint[:] mask) noexcept nogil:
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