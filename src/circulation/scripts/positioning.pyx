# distutils: language=c++
# cython: boundscheck=False, wraparound=False

"""
Extension class that provides the backend of the transform service
The TransformManager efficiently stores the velocity data (given the frequency they drop at, 
Python objects would fill the memory in no time), and provides the functions to extract
timed transforms from them much faster
"""

import numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos
from libcpp.vector cimport vector


cdef class TransformManager:
	cdef vector[double] time_buffer                       # Timestamps in seconds (with decimal part)
	cdef vector[(double, double, double)] linear_buffer   # Linear speeds along each axis at each timestamp
	cdef vector[(double, double, double)] angular_buffer  # Rotation speeds along each axis at each timestamp

	def __init__(self):
		pass

	def add_velocity(self, double time, (double, double, double) linear, (double, double, double) angular):
		"""Add a velocity to the buffers
		   - time    : double                   : Timestamp at which the velocity was captured, in seconds with decimal part
		   - linear  : (double, double, double) : Linear speed along axes X, Y, Z
		   - angular : (double, double, double) : Rotation speed about axes X, Y, Z
		"""
		self.time_buffer.push_back(time)
		self.linear_buffer.push_back(linear)
		self.angular_buffer.push_back(angular)

	def drop_velocity(self, double end_time):
		"""Drop all velocity data older than `end_time`
		   Old velocity data accumulates in memory and slows down the transform computations a bit,
		   so it’s better to clean them up as regularly as possible
		   - end_time : double : Timestamp before which velocity data must be dropped
		"""
		cdef size_t i
		# The timestamps are in chronological order, so just loop on them until we reach the first one to be kept
		for i in range(self.time_buffer.size()):
			if self.time_buffer[i] >= end_time:
				if i >= 1:
					# The range is [begin, end[, so the value as `i` is kept
					self.time_buffer.erase(self.time_buffer.begin(), self.time_buffer.begin() + i)
					self.linear_buffer.erase(self.linear_buffer.begin(), self.linear_buffer.begin() + i)
					self.angular_buffer.erase(self.angular_buffer.begin(), self.angular_buffer.begin() + i)
				break

	def get_map_transforms(self, double[:] start_times, double end_time):
		"""Get a batch of transform matrices of the tracked frame from timestamps `start_times` to timestamp `end_time`
		   - start_times : double[N] : Start timestamps in seconds with decimal part
		   - end_time    : double    : Target timestamp in seconds with decimal part
		<----------------- ndarray[N, 4, 4] : 4×4 3D homogeneous transform matrices from each timestamp in `start_times` to `end_time`
		"""
		# Can’t do anything without data
		assert start_times.shape[0] > 0
		assert self.time_buffer.size() > 0

		# Now, I know what you’re going to say : Yes, this is hands-up one of the worst-looking functions I have ever written
		# Cython ctuples don’t allow iteration in pure C, however they are still our best option in terms of memory usage,
		# so get used to it.

		# So, a bit of explanation
		# All this is made overly complex by the need to do things in batch
		# Sure, we could have a function to get a single transform for a single start time, and call it for each start time
		# But as each transform requires iterating over almost all velocity data, it would quickly get quite inefficient,
		# especially since the velocity data drops at high frequencies and may sometimes not be cleaned up for some time
		# Besides, this needs to be as fast as possible, because meanwhile the vehicle moves and the transform gets further and further from reality
		# So we compute all required transforms in one pass, at the price of quite an unwieldy code

		# To make things as "clean" and efficient as possible, we allocate accumulators,
		# in which we store the relevant computations from the velocity data
		# Those accumulators are of size [nstarts, 3], and we fill each row after the other
		# Basically, each row contains the movement (not speed) from its associated start timestamp, only to the next one
		# Then, at the end, we iterate through those in reverse to cumulate them and make all transforms from start_time to end_time

		cdef Py_ssize_t nstarts = start_times.shape[0]
		cdef double* linear_movement_ptr = <double*> malloc(nstarts * 3 * sizeof(double))
		cdef double* angular_movement_ptr = <double*> malloc(nstarts * 3 * sizeof(double))
		cdef double[:, :] linear_movement = <double[:nstarts, :3]> linear_movement_ptr
		cdef double[:, :] angular_movement = <double[:nstarts, :3]> angular_movement_ptr

		# Initialize the accumulators to 0
		cdef Py_ssize_t i, j
		for i in range(nstarts):
			for j in range(3):
				linear_movement[i, j] = 0
				angular_movement[i, j] = 0

		cdef double interval_duration, total_duration, t_start, t_end, base_time, next_time, slope_factor
		cdef (double, double, double) linear_start, linear_end, angular_start, angular_end

		cdef Py_ssize_t start_index = 0  # Current index among the start_times
		cdef size_t buffer_index = 0     # Current index in the velocity buffers

		# FIRST STEP : Some start timestamps may predate any of the available velocity data
		# For those, we just extrapolate from the first velocity data available

		linear_end = self.linear_buffer.front()
		angular_end = self.angular_buffer.front()

		# Iterate over intervals where both the current start timestamp and the next are before the earliest data available
		while start_index < start_times.shape[0] - 1 and start_times[start_index + 1] < self.time_buffer.front():
			interval_duration = start_times[start_index + 1] - start_times[start_index]
			linear_movement[start_index, 0] += interval_duration * linear_end[0]
			linear_movement[start_index, 1] += interval_duration * linear_end[1]
			linear_movement[start_index, 2] += interval_duration * linear_end[2]
			angular_movement[start_index, 0] += interval_duration * angular_end[0]
			angular_movement[start_index, 1] += interval_duration * angular_end[1]
			angular_movement[start_index, 2] += interval_duration * angular_end[2]
			start_index += 1

		# If only the last start_time remains and it predates the earliest available data
		if start_times[start_index] < self.time_buffer.front():
			interval_duration = self.time_buffer.front() - start_times[start_index]
			linear_movement[start_index, 0] += interval_duration * linear_end[0]
			linear_movement[start_index, 1] += interval_duration * linear_end[1]
			linear_movement[start_index, 2] += interval_duration * linear_end[2]
			angular_movement[start_index, 0] += interval_duration * angular_end[0]
			angular_movement[start_index, 1] += interval_duration * angular_end[1]
			angular_movement[start_index, 2] += interval_duration * angular_end[2]

		# SECOND STEP : Now we get to the actual velocity data buffers
		# At this point, the current start time is the first velocity timestamp or later
		# Now we loop over all data in the velocity buffers and accumulate it in the movement accumulators

		while buffer_index < self.time_buffer.size() - 1:
			base_time = self.time_buffer[buffer_index]
			next_time = self.time_buffer[buffer_index + 1]
			total_duration = next_time - base_time  # Duration of the whole time interval, to integrate the speed over it

			# There are quite a few edge conditions to check

			# First, if the current start_time is after next_time,
			# that time interval is completely irrelevant, so skip it
			if next_time <= start_times[start_index]:
				buffer_index += 1
				continue
			
			# If the current start time is between base_time and next_time, the first part of that time interval
			# is irrelevant, so start only at the start time
			if base_time <= start_times[start_index]:
				base_time = start_times[start_index]
			
			# If the end_time is between base_time and next_time, the latter part of that time interval is irrelevant,
			# so stop at end_time
			if next_time >= end_time:
				next_time = end_time
			
			# Sometimes, base_time and next_time (at this point) are equal
			# There are two possible reasons :
			# - One of the start_times and end_time are equal. In that case, we must skip this part altogether as the transform must be identity
			#   That’s the first condition (in that case both start_time and end_time are in-between base_time and next_time)
			# - Sometimes we get two velocity data for the same timestamp (± a few nanoseconds that get squished by the floating-point storage)
			#   In that case, we must just skip this time interval
			if base_time == next_time:
				if next_time >= end_time:
					break
				else:
					buffer_index += 1
					continue
			
			# The start_index might be a bit behind at this point, like when multiple start_times are equal or very close,
			# or when the previous step left it there. In that case, go to the next start_index and try again
			if start_index < start_times.shape[0] - 1 and base_time >= start_times[start_index + 1]:
				start_index += 1
				continue
			
			# Velocity values at the beginning and end of this time interval
			# From those, we perform a linear interpolation then integrate to get the actual position change
			# so this makes a quadratic interpolation of the actual movement
			linear_start = self.linear_buffer[buffer_index]
			linear_end = self.linear_buffer[buffer_index + 1]

			angular_start = self.angular_buffer[buffer_index]
			angular_end = self.angular_buffer[buffer_index + 1]

			# The next start_time is between base_time and next_time
			# In that case, we must cut in the middle to have the current accumulator end at the right time,
			# and the next accumulator start at that same time
			# So this is the same as the `else` part that does it over the whole interval, except it is cut in two
			if start_index < start_times.shape[0] - 1 and next_time >= start_times[start_index + 1]:
				interval_duration = start_times[start_index + 1] - base_time  # End at the next start time
				slope_factor = 1 / (2*total_duration)

				linear_movement[start_index, 0] += interval_duration * (linear_start[0] + (linear_end[0]-linear_start[0]) * slope_factor * interval_duration)
				linear_movement[start_index, 1] += interval_duration * (linear_start[1] + (linear_end[1]-linear_start[1]) * slope_factor * interval_duration)
				linear_movement[start_index, 2] += interval_duration * (linear_start[2] + (linear_end[2]-linear_start[2]) * slope_factor * interval_duration)
				angular_movement[start_index, 0] += interval_duration * (angular_start[0] + (angular_end[0]-angular_start[0]) * slope_factor * interval_duration)
				angular_movement[start_index, 1] += interval_duration * (angular_start[1] + (angular_end[1]-angular_start[1]) * slope_factor * interval_duration)
				angular_movement[start_index, 2] += interval_duration * (angular_start[2] + (angular_end[2]-angular_start[2]) * slope_factor * interval_duration)

				start_index += 1
				
				t_start = interval_duration  # Start after the first part of the interval
				t_end = next_time - base_time
				interval_duration = t_end - t_start

				linear_movement[start_index, 0] += interval_duration * (linear_start[0] + (linear_end[0]-linear_start[0]) * slope_factor * (t_start + t_end))
				linear_movement[start_index, 1] += interval_duration * (linear_start[1] + (linear_end[1]-linear_start[1]) * slope_factor * (t_start + t_end))
				linear_movement[start_index, 2] += interval_duration * (linear_start[2] + (linear_end[2]-linear_start[2]) * slope_factor * (t_start + t_end))
				angular_movement[start_index, 0] += interval_duration * (angular_start[0] + (angular_end[0]-angular_start[0]) * slope_factor * (t_start + t_end))
				angular_movement[start_index, 1] += interval_duration * (angular_start[1] + (angular_end[1]-angular_start[1]) * slope_factor * (t_start + t_end))
				angular_movement[start_index, 2] += interval_duration * (angular_start[2] + (angular_end[2]-angular_start[2]) * slope_factor * (t_start + t_end))
			# Otherwise, just integrate over the whole interval and add it to the accumulator
			# Those formulae are just to linearly interpolate between the values at base_time and next_time, then integrate over integral_duration to get the position
			else:
				interval_duration = next_time - base_time
				total_duration = next_time - base_time
				slope_factor = 1 / (2*total_duration)

				linear_movement[start_index, 0] += interval_duration * (linear_start[0] + (linear_end[0]-linear_start[0]) * slope_factor * interval_duration)
				linear_movement[start_index, 1] += interval_duration * (linear_start[1] + (linear_end[1]-linear_start[1]) * slope_factor * interval_duration)
				linear_movement[start_index, 2] += interval_duration * (linear_start[2] + (linear_end[2]-linear_start[2]) * slope_factor * interval_duration)
				angular_movement[start_index, 0] += interval_duration * (angular_start[0] + (angular_end[0]-angular_start[0]) * slope_factor * interval_duration)
				angular_movement[start_index, 1] += interval_duration * (angular_start[1] + (angular_end[1]-angular_start[1]) * slope_factor * interval_duration)
				angular_movement[start_index, 2] += interval_duration * (angular_start[2] + (angular_end[2]-angular_start[2]) * slope_factor * interval_duration)

			# The `end_time` was within this time interval : no need to continue any further, quit
			if next_time >= end_time:
				break
			else:
				buffer_index += 1

		# THIRD STEP : The `end_time` may be later than the latest available velocity data
		# In that case, we extrapolate based on the latest available velocity values

		if end_time > self.time_buffer.back():
			base_time = self.time_buffer.back()
			linear_start = self.linear_buffer.back()
			angular_start = self.angular_buffer.back()

			# Some start timestamps are after the last available velocity data, so extrapolate for each after the other
			# So first, if we still are not at the last start_time, the current start_time may be already started
			# so update it as necessary, then go forward with the next start times
			if start_index < start_times.shape[0] - 1:
				if start_times[start_index] > base_time:
					base_time = start_times[start_index]
				
				interval_duration = start_times[start_index + 1] - base_time

				linear_movement[start_index, 0] += interval_duration * linear_start[0]
				linear_movement[start_index, 1] += interval_duration * linear_start[1]
				linear_movement[start_index, 2] += interval_duration * linear_start[2]
				angular_movement[start_index, 0] += interval_duration * angular_start[0]
				angular_movement[start_index, 1] += interval_duration * angular_start[1]
				angular_movement[start_index, 2] += interval_duration * angular_start[2]

				start_index += 1

				# Extrapolate from one start_time to the next
				while start_index < start_times.shape[0] - 1:
					interval_duration = start_times[start_index + 1] - start_times[start_index]

					linear_movement[start_index, 0] += interval_duration * linear_start[0]
					linear_movement[start_index, 1] += interval_duration * linear_start[1]
					linear_movement[start_index, 2] += interval_duration * linear_start[2]
					angular_movement[start_index, 0] += interval_duration * angular_start[0]
					angular_movement[start_index, 1] += interval_duration * angular_start[1]
					angular_movement[start_index, 2] += interval_duration * angular_start[2]

					start_index += 1

				# Update the relevant start time
				base_time = start_times[start_index]
			
			# Now, base_time is set up as necessary for the current transform,
			# and we are finishing with the last start_time

			interval_duration = end_time - base_time

			linear_movement[start_index, 0] += interval_duration * linear_start[0]
			linear_movement[start_index, 1] += interval_duration * linear_start[1]
			linear_movement[start_index, 2] += interval_duration * linear_start[2]
			angular_movement[start_index, 0] += interval_duration * angular_start[0]
			angular_movement[start_index, 1] += interval_duration * angular_start[1]
			angular_movement[start_index, 2] += interval_duration * angular_start[2]

		# LAST STEP : Compute the actual transforms
		# At this point, we have accumulators with linear and angular movement from one start_time to the next,
		# and to end_time for the last one
		# Now we need to build actual transform matrices from those
		# So we iterate over the accumulator rows *in reverse* to cumulate them back
		# (the last transform is just from the last row, the previous one from the 2 last rows, ...)

		transforms = np.empty((nstarts, 4, 4))
		cdef double[:, :, :] transforms_view = transforms
		for start_index in range(nstarts-1, -1, -1):
			if start_index < nstarts - 1:
				# Sum with the row immediately after, as we iterate in reverse it already is accumulated over all following rows
				for i in range(3):
					linear_movement[start_index, i] += linear_movement[start_index + 1, i]
					angular_movement[start_index, i] += angular_movement[start_index + 1, i]

			# Intermediate results for the rotation part of the matrix
			cos_alpha = cos(angular_movement[start_index, 0])
			sin_alpha = sin(angular_movement[start_index, 0])
			cos_beta = cos(angular_movement[start_index, 1])
			sin_beta = sin(angular_movement[start_index, 1])
			cos_gamma = cos(angular_movement[start_index, 2])
			sin_gamma = sin(angular_movement[start_index, 2])

			# Rotation part of the matrix, from Euler angles
			transforms_view[start_index, 0, 0] = cos_beta*cos_gamma
			transforms_view[start_index, 0, 1] = sin_alpha*sin_beta*cos_gamma - cos_alpha*sin_gamma
			transforms_view[start_index, 0, 2] = cos_alpha*sin_beta*cos_gamma + sin_alpha*sin_gamma
			transforms_view[start_index, 1, 0] = cos_beta*sin_gamma
			transforms_view[start_index, 1, 1] = sin_alpha*sin_beta*sin_gamma + cos_alpha*cos_gamma
			transforms_view[start_index, 1, 2] = cos_alpha*sin_beta*sin_gamma - sin_alpha*cos_gamma
			transforms_view[start_index, 2, 0] = -sin_beta
			transforms_view[start_index, 2, 1] = sin_alpha*cos_beta
			transforms_view[start_index, 2, 2] = cos_alpha*cos_beta

			# Translation part
			transforms_view[start_index, 0, 3] = -linear_movement[start_index, 0]
			transforms_view[start_index, 1, 3] = -linear_movement[start_index, 1]
			transforms_view[start_index, 2, 3] = -linear_movement[start_index, 2]

			# And affine part
			transforms_view[start_index, 3, 0] = 0
			transforms_view[start_index, 3, 1] = 0
			transforms_view[start_index, 3, 2] = 0
			transforms_view[start_index, 3, 3] = 1

		free(linear_movement_ptr)
		free(angular_movement_ptr)

		return transforms