import numpy as np

cimport cython
from libc.math cimport sqrt

# Return value of _project_on_curve_index
cdef struct _ProjectionIndex:
	int index
	double vector_factor

# Return value of _next_point
cdef struct _NextPoint:
	double x, y
	int index
	double vector_factor


cpdef double vector_angle(double[:] v1, double[:] v2)
cpdef double line_length(double[:, :] line)
cdef _ProjectionIndex _project_on_curve_index(double point_x, double point_y, double[:, :] curve, bint extend, Py_ssize_t min_index)
cdef _NextPoint _next_point(Py_ssize_t closest_index, double vector_factor, double[:, :] curve, double target_distance)
cdef (_NextPoint, double) _next_point_score(double[:] point, double[:, :] curve, double[:] scores, double target_distance, bint extend, Py_ssize_t min_index)
cpdef bint segments_intersect(double[:] A, double[:] B, double[:] C, double[:] D)
cpdef void find_self_intersects(double[:, :] curve, bint[:] mask)
cpdef int savgol_window(int base_size, int array_size)
cdef void _savgol_coeffs_degree23(double[:] coeffs, int window_size)