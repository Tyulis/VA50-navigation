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
cdef (_NextPoint, double) _next_point_score(Py_ssize_t closest_index, double vector_factor, double[:, :] curve, double[:] scores, double target_distance, bint extend, Py_ssize_t min_index)
cpdef _ProjectionIndex project_from(double[:] point, double[:] vector, double[:, :] target_curve, bint extend, Py_ssize_t min_index)
cpdef bint segments_intersect(double[:] A, double[:] B, double[:] C, double[:] D)
cpdef void find_self_intersects(double[:, :] curve, bint[:] mask)
cpdef int savgol_window(int base_size, int array_size)
cdef void _savgol_coeffs_degree23(double[:] coeffs, int window_size)