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
Fuzzy logic module, used to choose lane markings
This is basic fuzzy logic, except it takes 2D variables as it’s for pairs of curves,
and it uses malus to infer the ruleset, for easy parameterization
Basically, every rule has a base score, and each level of each variable is associated to a malus applied to that score
The output value of that rule is the result
"""

import numpy as np
cimport cython
from libc.stdlib cimport malloc, free


cdef class FuzzySystem:
	cdef Py_ssize_t num_variables  # Number of input variables
	cdef Py_ssize_t num_sets       # Number of fuzzy sets for each variable
	cdef Py_ssize_t num_outputs    # Number of output sets
	cdef Py_ssize_t num_rules      # Total number of rules

	cdef int* ruleset
	cdef int* rulegroup_counts
	cdef double* centers_ptr
	cdef double[:, ::1] centers
	cdef double* output_centers

	def __init__(self, double[:, :] centers, long[:, :] malus, double[:] output_centers, long base_score):
		"""Initialize the fuzzy system and precompute the ruleset
		   V = number of variables, S = number of sets per variable, C = number of output sets
		   - centers        : double[V, S] : Centers for each set of each variable
		   - malus          : long[V, S]   : Malus applied to the base score for each set of each variable
		   - output_centers : long[C]      : Center for each set of the output variable
		   - base_score     : long         : Base score for the ruleset, translates to output variables sets after malus
		"""
		self.num_variables = centers.shape[0]
		self.num_sets = centers.shape[1]
		self.num_outputs = output_centers.shape[0]
		self.num_rules = self.num_sets**self.num_variables

		assert self.num_sets == 3

		# Allocate the arrays
		self.ruleset = <int*>malloc(self.num_rules * sizeof(int))
		self.rulegroup_counts = <int*>malloc(self.num_outputs * sizeof(int))
		self.centers_ptr = <double*>malloc(self.num_variables * self.num_sets * sizeof(double))
		self.centers = <double[:self.num_variables, :self.num_sets:1]> self.centers_ptr
		self.output_centers = <double*>malloc(self.num_outputs * sizeof(double))

		# Copy the centers and output_centers arrays, initialize the rulegroup_counts accumulator by the bay
		cdef Py_ssize_t i, j
		for i in range(self.num_variables):
			for j in range(self.num_sets):
				self.centers[i, j] = centers[i, j]
		for i in range(self.num_outputs):
			self.output_centers[i] = output_centers[i]
			self.rulegroup_counts[i] = 0

		# Precompute the ruleset
		# Each rule is identified by a unique index built from the sets of each variable that lead to it
		# For example, with 3 sets per variables, it would be something like 10122 in base 3 for sets 1, 0, 1, 2, and 2 of the respective variables
		# As such, it can accomodate any amount of variables with a single dimension, no need for arbitrary dimensions 
		cdef Py_ssize_t rule_index, temp_index, var_index, set_index
		cdef long score
		for rule_index in range(self.num_rules):
			score = base_score
			
			# Decompose the index into the respective set indices for each variable, and add the corresponding malus 
			temp_index = rule_index
			for var_index in range(self.num_variables):
				set_index = temp_index % self.num_sets
				temp_index //= self.num_sets
				score += malus[var_index, set_index]
			
			# Clip at 0
			if score < 0:
				score = 0
			
			# Set the rule, and also update the counts (number of rules that lead to each output set)
			self.ruleset[rule_index] = score
			self.rulegroup_counts[score] += 1

	def __dealloc__(self):
		"""Free the malloc-allocated arrays"""
		free(self.ruleset)
		free(self.rulegroup_counts)
		free(self.centers_ptr)
		free(self.output_centers)		

	cdef void _fuzzify(self, double[:, :, :] variables, double[:, :, :, ::1] set_functions) noexcept:
		"""Fuzzify the variables according the the set centers
		   - variables     : double[V, y, x]    : Values of each variable V for each input pair (y, x)
		   - set_functions : double[V, S, y, x] : Output, fuzzy membership coefficient of each variable V to each set S for each input pair (y, x)
		"""
		cdef Py_ssize_t y_size = variables.shape[1], x_size = variables.shape[2]
		cdef Py_ssize_t var_index, set_index, x, y
		cdef bint reverse
		cdef double var_value
		cdef double[:] var_centers

		for var_index in range(self.num_variables):
			var_centers = self.centers[var_index]

			# Check whether the centers are reversed (higher is better instead of lower is better), as it flips the whole logic
			# The convention as the output of this function is best first, worst last, regardless of the input logic
			reverse = var_centers[0] > var_centers[1]
			for y in range(y_size):
				for x in range(x_size):
					var_value = variables[var_index, y, x]

					if not reverse:
						# Before the first center -> 100% good
						if var_value < var_centers[0]:
							set_functions[var_index, 0, y, x] = 1
							set_functions[var_index, 1, y, x] = 0
							set_functions[var_index, 2, y, x] = 0
						# Between the first and second centers -> triangular good/medium
						elif var_value < var_centers[1]:
							set_functions[var_index, 1, y, x] = (var_value - var_centers[0]) / (var_centers[1] - var_centers[0])
							set_functions[var_index, 0, y, x] = 1 - set_functions[var_index, 1, y, x]
							set_functions[var_index, 2, y, x] = 0
						# Between the second and third centers -> triangular medium/bad
						elif var_value < var_centers[2]:
							set_functions[var_index, 0, y, x] = 0
							set_functions[var_index, 2, y, x] = (var_value - var_centers[1]) / (var_centers[2] - var_centers[1])
							set_functions[var_index, 1, y, x] = 1 - set_functions[var_index, 2, y, x]
						# After the last center -> 100% bad
						else:
							set_functions[var_index, 0, y, x] = 0
							set_functions[var_index, 1, y, x] = 0
							set_functions[var_index, 2, y, x] = 1
					# Reversed : Higher is better, indices are reversed
					else:
						# After the last center -> 100% bad
						if var_value < var_centers[2]:
							set_functions[var_index, 2, y, x] = 1
							set_functions[var_index, 1, y, x] = 0
							set_functions[var_index, 0, y, x] = 0
						# Between the third and second centers -> triangular medium/bad
						elif var_value < var_centers[1]:
							set_functions[var_index, 1, y, x] = (var_value - var_centers[2]) / (var_centers[1] - var_centers[2])
							set_functions[var_index, 2, y, x] = 1 - set_functions[var_index, 1, y, x]
							set_functions[var_index, 0, y, x] = 0
						# Between the first and second centers -> triangular good/medium
						elif var_value < var_centers[0]:
							set_functions[var_index, 2, y, x] = 0
							set_functions[var_index, 0, y, x] = (var_value - var_centers[1]) / (var_centers[0] - var_centers[1])
							set_functions[var_index, 1, y, x] = 1 - set_functions[var_index, 0, y, x]
						# Before the first center -> 100% good
						else:
							set_functions[var_index, 2, y, x] = 0
							set_functions[var_index, 1, y, x] = 0
							set_functions[var_index, 0, y, x] = 1

	def fuzzy_best(self, double[:, :, :] variables):
		"""Retrieve the best among all given pairs
		   - variables : double[V, y, x] : Value of each input variable for each input pair
		<--------------- int             : Y index (first index) of the best pair
		<--------------- int             : X index (second index) of the best pair
		<--------------- double          : Score of the best pair
		"""
		cdef Py_ssize_t y_size = variables.shape[1], x_size = variables.shape[2]
		cdef double* set_functions_ptr = <double*>malloc(self.num_variables * self.num_sets * y_size * x_size * sizeof(double))
		cdef double[:, :, :, ::1] set_functions = <double[:self.num_variables, :self.num_sets, :y_size, :x_size:1]> set_functions_ptr

		# Get the membership coefficients for each variable, each set and each pair
		# Glory to Cython’s multidimensional memoryviews
		self._fuzzify(variables, set_functions)

		# Now infer with sum-product weighted heights inference and keep the pair with the highest score
		# In the `set_functions` set dimension, lower index is better
		cdef double* conditions = <double*>malloc(self.num_outputs * sizeof(double))
		cdef double condition, numerator, denominator, score
		cdef double best_score = 0
		cdef Py_ssize_t rule_index, temp_index, j, best_y = -1, best_x = -1
		for y in range(y_size):
			for x in range(x_size):
				# We need to get the condition coefficient for each output center,
				# So start at 0 and calculate the mean over all rules in the ruleset for each pair
				# Thankfully, this can be done in pure C
				for j in range(self.num_outputs):
					conditions[j] = 0

				# Same logic as before, the rules’ indices are their input fuzzy sets encoded in base 3
				for rule_index in range(self.num_rules):
					condition = 1
					temp_index = rule_index
					for var_index in range(self.num_variables):
						set_index = temp_index % self.num_sets
						temp_index //= self.num_sets
						condition *= set_functions[var_index, set_index, y, x]
					conditions[self.ruleset[rule_index]] += condition
				
				# Here come the precomputed counts for the mean
				numerator = 0
				denominator = 0
				for j in range(self.num_outputs):
					numerator += (conditions[j] / self.rulegroup_counts[j]) * self.output_centers[j]
					denominator += conditions[j] / self.rulegroup_counts[j]
				score = numerator / denominator

				# Keep the max
				if score > best_score:
					best_score = score
					best_y = y
					best_x = x
		free(set_functions_ptr)
		free(conditions)
		return best_y, best_x, best_score

	def fuzzy_scores(self, double[:, :, :] variables):
		"""Get the score associated to each input pair
		   - variables : double[V, y, x] : Value of each input variable for each input pair
		<--------------- double[y, x]    : Scores of each input pair
		"""
		# This does exactly the same thing as `fuzzy_best`, except it stores all scores instead of just keeping the max
		cdef Py_ssize_t y_size = variables.shape[1], x_size = variables.shape[2]
		cdef double* set_functions_ptr = <double*>malloc(self.num_variables * self.num_sets * y_size * x_size * sizeof(double))
		cdef double[:, :, :, ::1] set_functions = <double[:self.num_variables, :self.num_sets, :y_size, :x_size:1]> set_functions_ptr

		self._fuzzify(variables, set_functions)

		scores = np.empty((y_size, x_size))
		cdef double[:, :] score_view = scores
		cdef double* conditions = <double*>malloc(self.num_outputs * sizeof(double))
		cdef double condition, numerator, denominator, score
		cdef double best_score = 0
		cdef Py_ssize_t rule_index, temp_index, j, best_y = -1, best_x = -1
		for y in range(y_size):
			for x in range(x_size):
				for j in range(self.num_outputs):
					conditions[j] = 0

				for rule_index in range(self.num_rules):
					condition = 1
					temp_index = rule_index
					for var_index in range(self.num_variables):
						set_index = temp_index % self.num_sets
						temp_index //= self.num_sets
						condition *= set_functions[var_index, set_index, y, x]
					conditions[self.ruleset[rule_index]] += condition
				
				numerator = 0
				denominator = 0
				for j in range(self.num_outputs):
					numerator += (conditions[j] / self.rulegroup_counts[j]) * self.output_centers[j]
					denominator += conditions[j] / self.rulegroup_counts[j]
				score_view[y, x] = numerator / denominator
		free(set_functions_ptr)
		free(conditions)
		return scores