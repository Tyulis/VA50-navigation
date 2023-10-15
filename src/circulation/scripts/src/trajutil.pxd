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

from libcpp.vector cimport vector

cdef bint pca_2x2(double[:, :] points, double[2] eigenvalues, double[2][2] eigenvectors) noexcept
cdef void inverse_2x2(double[2][2] matrix, double[2][2] inverse) noexcept nogil
cdef void cluster_DBSCAN(double[:, :] data, long[:] labels, double epsilon, long min_samples)
cdef vector[Py_ssize_t] compact_array(double[:, :] array, bint[:] mask) noexcept nogil