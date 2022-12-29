# distutils: language=c++
# cython: boundscheck=False, wraparound=False

"""
Extension class that provides the backend of the transform service
The TransformManager efficiently stores the velocity data (given the frequency they drop at, 
Python objects would fill the memory in no time), and provides the functions to extract
timed transforms from them much faster

Terminology note :
	- Everything referred to as `time periods` is the time intervals between velocity data points,
	  to avoid confusion with immediate time points or intervals
"""

import numpy as np
cimport cython
from cython.operator cimport dereference as deref, postincrement as postinc
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport sqrt
from vector11 cimport vector  # We need the C++11 version to get the reverse iterators

# Cython’s libcpp doesn’t have the <array> header yet, so let’s make this ourselves
cdef struct Vector3:
	double x
	double y
	double z

cdef struct Quaternion:
	double w
	double x
	double y
	double z


cdef class TransformManager:
	cdef double sim_interval
	cdef vector[double] time_buffer         # Timestamps in seconds (with decimal part)
	cdef vector[Vector3] linear_buffer      # Linear speeds along each axis at each timestamp
	cdef vector[Quaternion] angular_buffer  # Rotation speeds along each axis at each timestamp

	def __init__(self, double sim_interval):
		"""Initialize the transform manager
		   - sim_interval : double : Interval at which the angular velocity gets interpolated, in seconds
		                             Shorter is better but slower"""
		self.sim_interval = sim_interval

	def add_velocity(self, double time, (double, double, double) linear, (double, double, double) angular):
		"""Add a velocity to the buffers
		   - time    : double                   : Timestamp at which the velocity was captured, in seconds with decimal part
		   - linear  : (double, double, double) : Linear speed along axes X, Y, Z
		   - angular : (double, double, double) : Rotation speed about axes X, Y, Z
		"""
		self.time_buffer.push_back(time)
		self.linear_buffer.push_back(Vector3(x=linear[0], y=linear[1], z=linear[2]))
		self.angular_buffer.push_back(Quaternion(w=0, x=angular[0], y=angular[1], z=angular[2]))

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

		# These buffers store the translation vectors and rotation quaternions for each element of start_times
		cdef Py_ssize_t nstarts = start_times.shape[0]
		cdef Vector3* translation = <Vector3*> malloc(nstarts * sizeof(Vector3))
		cdef Quaternion* rotation = <Quaternion*> malloc(nstarts * sizeof(Quaternion))

		cdef double current_time = end_time
		cdef Py_ssize_t start_index = start_times.shape[0] - 1

		# Initialize the transformation
		translation[start_index] = Vector3(x=0, y=0, z=0)
		rotation[start_index] = Quaternion(w=1, x=0, y=0, z=0)

		# Use const reverse iterators, as we iterate over those in reverse
		cdef vector[double].const_reverse_iterator time_iterator = self.time_buffer.crbegin()
		cdef vector[Vector3].const_reverse_iterator linear_iterator = self.linear_buffer.crbegin()
		cdef vector[Quaternion].const_reverse_iterator angular_iterator = self.angular_buffer.crbegin()

		# The end time may be after the last known velocity data
		# In that case, extrapolate the necessary data with the last angle velocity
		while current_time > deref(time_iterator):
			# The current start time is after the last known velocity data, extrapolate and go to the next start time
			if start_times[start_index] > deref(time_iterator):
				self.extrapolate_linear_velocity_reverse(&translation[start_index], deref(linear_iterator), current_time - start_times[start_index])
				self.integrate_angular_velocity_reverse_direct(&rotation[start_index], deref(angular_iterator), current_time - start_times[start_index])
				current_time = start_times[start_index]
				start_index = self.next_start_time(translation, rotation, start_index)
			# The current start time is before the last known velocity data, extrapolate that part and go to the interpolation part
			else:
				self.extrapolate_linear_velocity_reverse(&translation[start_index], deref(linear_iterator), current_time - deref(time_iterator))
				self.integrate_angular_velocity_reverse_direct(&rotation[start_index], deref(angular_iterator), current_time - deref(time_iterator))
				current_time = deref(time_iterator)
			
			if start_index < 0:
				break
		
		# Now, we are within the time range we’ve got data about, so interpolate and integrate the velocity data between our start points
		# To keep track of which is which, we keep those in order (end is the one after start),
		# but keep in mind we integrate in reverse order, end -> start
		cdef Vector3 linear_start, linear_end
		cdef Quaternion angular_start, angular_end
		cdef double period_start, period_end

		period_start = deref(time_iterator)
		linear_start = deref(linear_iterator)
		angular_start = deref(angular_iterator)

		postinc(time_iterator)
		postinc(linear_iterator)
		postinc(angular_iterator)

		while linear_iterator != self.linear_buffer.crend() and start_index >= 0:
			period_end = period_start
			linear_end = linear_start
			angular_end = angular_start

			period_start = deref(time_iterator)
			linear_start = deref(linear_iterator)
			angular_start = deref(angular_iterator)

			postinc(time_iterator)
			postinc(linear_iterator)
			postinc(angular_iterator)

			# Now a few edge conditions to check
			# Sometimes we get velocity data for intervals of a few nanoseconds that get squished by the floating-point representation,
			# skip them to avoid zero-division issues
			if period_start == period_end:
				continue

			# If this time interval is entirely after the end_time, no need to dwell on it
			if period_start > end_time:
				continue

			# When we enter this loop, current_time is end_time or the last velocity data timestamp,
			# so either way, no need to worry about it as values will be integrated until current_time

			# Handle the start_times that are in this time period
			while start_times[start_index] > period_start and start_index >= 0:
				self.integrate_linear_velocity_reverse(&translation[start_index], linear_start, linear_end, period_start, period_end, start_times[start_index], current_time)
				self.integrate_angular_velocity_reverse(&rotation[start_index], angular_start, angular_end, period_start, period_end, start_times[start_index], current_time)

				current_time = start_times[start_index]
				start_index = self.next_start_time(translation, rotation, start_index)
			if start_index < 0:
				break
						
			# Then the one that goes through to the previous time period
			self.integrate_linear_velocity_reverse(&translation[start_index], linear_start, linear_end, period_start, period_end, period_start, current_time)
			self.integrate_angular_velocity_reverse(&rotation[start_index], angular_start, angular_end, period_start, period_end, period_start, current_time)
			current_time = period_start

		# Now, we are just before the beginning of our velocity data
		# Some start_times might be before it, so we also need to extrapolate in that direction
		while start_index >= 0:
			self.extrapolate_linear_velocity_reverse(&translation[start_index], linear_start, current_time - start_times[start_index])
			self.integrate_angular_velocity_reverse_direct(&rotation[start_index], angular_start, current_time - start_times[start_index])
			current_time = start_times[start_index]
			start_index = self.next_start_time(translation, rotation, start_index)
		
		# At this point, we have stored all of our transformations in the linear and angular transformation buffers
		# Now, let’s build transformation matrices out of them

		transforms = np.empty((nstarts, 4, 4))
		cdef double[:, :, :] transforms_view = transforms
		cdef double s = 1  # TODO : Check whether all of our quaternions are unitary
		cdef Py_ssize_t i
		cdef Quaternion current_rotation
		for i in range(nstarts):
			# Formula for the rotation matrix from a quaternion from here :
			# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
			# We normalize the quaternion at this step
			current_rotation = rotation[i]
			s = 1/(current_rotation.w*current_rotation.w + current_rotation.x*current_rotation.x +
			       current_rotation.y*current_rotation.y + current_rotation.z*current_rotation.z)
			# Rotation matrix from the resulting quaternion
			# We are in reverse, so this is the transposed matrix
			transforms_view[i, 0, 0] = 1 - 2*s * (current_rotation.y**2 + current_rotation.z**2)
			transforms_view[i, 1, 0] = 2*s * (current_rotation.x * current_rotation.y - current_rotation.z * current_rotation.w)
			transforms_view[i, 2, 0] = 2*s * (current_rotation.x * current_rotation.z + current_rotation.y * current_rotation.w)
			transforms_view[i, 0, 1] = 2*s * (current_rotation.x * current_rotation.y + current_rotation.z * current_rotation.w)
			transforms_view[i, 1, 1] = 1 - 2*s * (current_rotation.x**2 + current_rotation.z**2)
			transforms_view[i, 2, 1] = 2*s * (current_rotation.y * current_rotation.z - current_rotation.x * current_rotation.w)
			transforms_view[i, 0, 2] = 2*s * (current_rotation.x * current_rotation.z - current_rotation.y * current_rotation.w)
			transforms_view[i, 1, 2] = 2*s * (current_rotation.y * current_rotation.z + current_rotation.x * current_rotation.w)
			transforms_view[i, 2, 2] = 1 - 2*s * (current_rotation.x**2 + current_rotation.y**2)

			# Translation vector
			transforms_view[i, 0, 3] = translation[i].x
			transforms_view[i, 1, 3] = translation[i].y
			transforms_view[i, 2, 3] = translation[i].z

			# Affine part
			transforms_view[i, 3, 0] = transforms_view[i, 3, 1] = transforms_view[i, 3, 2] = 0
			transforms_view[i, 3, 3] = 1

		free(translation)
		free(rotation)
		return transforms
	
	cdef void extrapolate_linear_velocity_reverse(self, Vector3* translation, Vector3 linear_speed, double timedelta):
		"""Extrapolate the linear velocity with the given speed over some time interval
		   - translation  : Vector3* : Translation vector to update
		   - linear_speed : Vector3  : Linear speed to apply
		   - timedelta    : double   : Duration over which to apply the linear speed
		"""
		translation.x -= linear_speed.x * timedelta
		translation.y -= linear_speed.y * timedelta
		translation.z -= linear_speed.z * timedelta
		
	cdef void integrate_linear_velocity_reverse(self, Vector3* translation, Vector3 linear_start, Vector3 linear_end, double period_start, double period_end, double start_time, double end_time):
		"""Integrate the linear velocity to compute the change in position between two time points
		   As we need everything from end to start, the result is reversed
		   - translation  : Vector3* : Translation vector to update
		   - linear_start : Vector3  : Linear speed at the beginning of the time period
		   - linear_end   : Vector3  : Linear speed at the end of the time period
		   - period_start : double   : Start of the velocity data time period
		   - period_end   : double   : End of the velocity data time period
		   - start_time   : double   : Time point at which to start the integration
		   - end_time     : double   : Time point at which to end the integration
		"""
		# We have the linear speed at the beginning and at the end of the time period,
		# so we’ve got initial speed and constant acceleration
		# We can thus integrate this into a quadratic formula for the position
		# As always, we are in reverse so -=
		# It *is* (end_time + start_time) next to the acceleration, it’s because here (end_time² - start_time²) got factored into (end_time - start_time)(end_time + start_time)
		# and we need to subtract 2*period_start as it needs to be relative to period_start
		cdef Vector3 save = Vector3(x=translation.x, y=translation.y, z=translation.z)
		translation.x -= (end_time - start_time) * (linear_start.x + (end_time + start_time - 2*period_start) * (linear_end.x-linear_start.x) / (2*(period_end - period_start)))
		translation.y -= (end_time - start_time) * (linear_start.y + (end_time + start_time - 2*period_start) * (linear_end.y-linear_start.y) / (2*(period_end - period_start)))
		translation.z -= (end_time - start_time) * (linear_start.z + (end_time + start_time - 2*period_start) * (linear_end.z-linear_start.z) / (2*(period_end - period_start)))
	

	# Now that’s the tricky part : integrating the linear velocity is trivial, but the angular velocity is on a whole other level
	# From M. Boyle, The integration of angular velocity, 2017, the angular velocity can be integrated using the following differential equation :
	# Ṙ(t) = 1/2 ω(t)R(t), where R(t) is the quaternion representing the rotation from the world frame to the body frame, and
	# ω(t) the angular velocity vector as a pure quaternion
	# Now, this is a system of non-linear differential equations and I’m no quaternion expert,
	# so let’s approximate it numerically using Taylor expansion and linear interpolation over small time steps
	# instead of solving this whole contraption analytically

	cdef void integrate_angular_velocity_reverse_direct(self, Quaternion* rotation, Quaternion angular_speed, double timedelta):
		"""Integrate a constant angular velocity over some time interval
		   - rotation      : Quaternion* : Rotation quaternion to update
		   - angular_speed : Quaternion  : Angular speed to apply
		   - timedelta     : double      : Time interval to integrate
		"""
		# See this : https://math.stackexchange.com/questions/189185/quaternion-differentiation
		# Note that the angular speed is first stored in chronological order, but as we need the transform from the end to the start,
		# we need it in reverse, hence the minus sign instead of +
		rotation.w -= (timedelta / 2) * (-rotation.x * angular_speed.x - rotation.y * angular_speed.y - rotation.z * angular_speed.z)
		rotation.x -= (timedelta / 2) * ( rotation.w * angular_speed.x + rotation.z * angular_speed.y - rotation.y * angular_speed.z)
		rotation.y -= (timedelta / 2) * (-rotation.z * angular_speed.x + rotation.w * angular_speed.y + rotation.x * angular_speed.z)
		rotation.z -= (timedelta / 2) * ( rotation.y * angular_speed.x - rotation.x * angular_speed.y + rotation.w * angular_speed.z)

		# Actually, normalizing the quaternion at intermediate steps doesn’t do us any significant good,
		# so let’s just leave it as is and normalize using the `s` factor when building the matrix
		# cdef double norm = sqrt(rotation.w*rotation.w + rotation.x*rotation.x + rotation.y*rotation.y + rotation.z*rotation.z)
		# rotation.w /= norm
		# rotation.x /= norm
		# rotation.y /= norm
		# rotation.z /= norm
	
	cdef void integrate_angular_velocity_reverse(self, Quaternion* rotation, Quaternion angular_start, Quaternion angular_end, double period_start, double period_end, double start_time, double end_time):
		"""Integrate a linearly interpolated angular velocity over time
		   - rotation      : Quaternion* : Rotation quaternion to update
		   - angular_start : Quaternion  : Angular velocity at the beginning of the period
		   - angular_end   : Quaternion  : Angular velocity at the end of the time period
		   - period_start  : double      : Start of the time period
		   - period_end    : double      : End of the time period
		   - start_time    : double      : Time point at which to start the integration
		   - end_time      : double      : Time point at which to end the integration
		"""
		cdef double current_time = end_time, timedelta
		cdef Quaternion current_angular_velocity
		while current_time > start_time:
			# Get the time delta over which we’re going to integrate as constant angular velocity
			if current_time - start_time < self.sim_interval:
				timedelta = current_time - start_time
			else:
				timedelta = self.sim_interval
			
			# Now get the linearly interpolated angular velocity
			current_angular_velocity = Quaternion(w = 0,
			                                      x = (current_time - period_start) * (angular_end.x - angular_start.x) / (period_end - period_start) + angular_start.x,
												  y = (current_time - period_start) * (angular_end.y - angular_start.y) / (period_end - period_start) + angular_start.y,
												  z = (current_time - period_start) * (angular_end.z - angular_start.z) / (period_end - period_start) + angular_start.z)
			# And integrate over this time step
			self.integrate_angular_velocity_reverse_direct(rotation, current_angular_velocity, timedelta)
			current_time -= timedelta
			

	cdef Py_ssize_t next_start_time(self, Vector3* translation, Quaternion* rotation, Py_ssize_t start_index):
		"""Convenience function to go to the next start time in the list :
		   Copy the current transform to the next index and return the next start index
		   - translation : Vector3[nstarts]    : Array of translations
		   - rotation    : Quaternion[nstarts] : Array of rotations
		   - start_index : Py_ssize_t          : Current start index. If it is 0, do not copy the transform and return a negative index
		<----------------- Py_ssize_t : Next start index
		"""
		# Copy if there is a next start time
		if start_index > 0:
			memcpy(&translation[start_index - 1], &translation[start_index], sizeof(Vector3))
			memcpy(&rotation[start_index - 1], &rotation[start_index], sizeof(Quaternion))
		return start_index - 1