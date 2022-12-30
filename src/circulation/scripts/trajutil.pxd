from libcpp.vector cimport vector

cdef bint pca_2x2(double[:, :] points, double[2] eigenvalues, double[2][2] eigenvectors) noexcept
cdef void inverse_2x2(double[2][2] matrix, double[2][2] inverse) noexcept nogil
cdef void cluster_DBSCAN(double[:, :] data, long[:] labels, double epsilon, long min_samples)
cdef vector[Py_ssize_t] compact_array(double[:, :] array, bint[:] mask) noexcept nogil