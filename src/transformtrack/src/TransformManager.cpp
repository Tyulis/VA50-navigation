#include "transformtrack/TransformManager.h"
#include <iostream>

// NOTE : Quaternions here are stored in order (w, x, y, z)

TransformManager::TransformManager() {

}

/** Initialize the transform manager
  * - sim_interval : double : Interval at which the angular velocity gets interpolated, in seconds
		                      Shorter is better but slower. Leaving this higher than 0.01 is likely to lead to very poor results */
TransformManager::TransformManager(double sim_interval) : m_sim_interval(sim_interval) {
	m_linear_buffer.set_size(3, 0);
	m_angular_buffer.set_size(3, 0);
}

/** Add a velocity to the buffers
  * - time    : double        : Timestamp at which the velocity was captured, in seconds with decimal part
  * - linear  : arma::dvec[3] : Linear speed along axes X, Y, Z
  * - angular : arma::dvec[3] : Rotation speed about axes X, Y, Z */
void TransformManager::add_velocity(double timestamp, arma::dvec linear, arma::dvec angular) {
	m_time_buffer.push_back(timestamp);
	m_linear_buffer.insert_cols(m_linear_buffer.n_cols, linear);
	m_angular_buffer.insert_cols(m_angular_buffer.n_cols, angular);
}

/** Drop all velocity data older than `end_time`
  * Old velocity data accumulates in memory, so it’s better to clean it up from time to time
  * - end_time : double : Timestamp before which velocity data must be dropped */
void TransformManager::drop_velocity(double end_time) {
	for (int i = 0; i < m_time_buffer.size(); i++) {
		if (m_time_buffer[i] >= end_time) {
			if (i >= 1) {
				// The range is [begin, end[, so the value at `i` is kept
				m_time_buffer.erase(m_time_buffer.begin(), m_time_buffer.begin() + i);
				m_linear_buffer.shed_cols(0, i - 1);
				m_angular_buffer.shed_cols(0, i - 1);
			}
			break;
		}
	}
}

/** Get a batch of transform matrices of the tracked frame from timestamps `start_times` to timestamp `end_time`
  * Note that the whole module operates on doubles but returns floats
  * The modules that use this operate in single precision, but as this is doing numerical integration,
  * we need the additional precision internally because the numerical stability of the whole thing is awful
  * - start_times : arma::dvec[N]        : Start timestamps in seconds with decimal part
  * - end_time    : double               : Target timestamp in seconds with decimal part
  * <-------------- arma::fcube[N, 4, 4] : 4×4 3D homogeneous transform matrices from each timestamp in `start_times` to `end_time` */
arma::fcube TransformManager::get_map_transforms(arma::dvec const& start_times, double end_time) {
	// Can’t do anything without data
	assert(start_times.n_elem > 0);
	assert(m_time_buffer.size() > 0);

	// Buffers to store the translation vectors and rotation quaternions for each element of `start_times`
	int num_starts = start_times.n_elem;
	arma::dmat translation(3, num_starts, arma::fill::zeros);
	arma::dmat rotation(4, num_starts, arma::fill::zeros);
	rotation.row(0).fill(1.0); // Start with w = 1, x = 0, y = 0, z = 0 quaternions

	double current_time = end_time;
	int start_index = num_starts - 1;
	int buffer_index = m_time_buffer.size() - 1;  // We iterate in reverse

	// The end time may be after the last known velocity data
	// In that case, extrapolate the necessary data with the last angle velocity
	while (current_time > m_time_buffer[buffer_index]) {
		// The current start time is after the last known velocity data, extrapolate and go to the next start time
		if (start_times[start_index] > m_time_buffer[buffer_index]) {
			extrapolate_linear_velocity_reverse(translation.col(start_index), m_linear_buffer.col(buffer_index), current_time - start_times[start_index]);
			integrate_angular_velocity_reverse_direct(rotation.col(start_index), m_angular_buffer.col(buffer_index), current_time - start_times[start_index]);
			current_time = start_times[start_index];
			start_index = next_start_time(translation, rotation, start_index);
		}

		// The current start time is before the last known velocity data, extrapolate that part and go to the interpolation part
		else {
			extrapolate_linear_velocity_reverse(translation.col(start_index), m_linear_buffer.col(buffer_index), current_time - m_time_buffer[buffer_index]);
			integrate_angular_velocity_reverse_direct(rotation.col(start_index), m_angular_buffer.col(buffer_index), current_time - m_time_buffer[buffer_index]);
			current_time = m_time_buffer[buffer_index];
		}

		if (start_index < 0)
			break;
	}

	// Now, we are within the time range we’ve got data about, so interpolate and integrate the velocity data between our start points
	// To keep track of which is which, we keep those in order (end is the one after start),
	// but keep in mind we integrate in reverse order, end -> start
	double period_start = m_time_buffer[buffer_index], period_end;
	arma::dvec linear_start = m_linear_buffer.col(buffer_index), linear_end;
	arma::dvec angular_start = m_angular_buffer.col(buffer_index), angular_end;
	buffer_index -= 1;

	while (buffer_index >= 0 && start_index >= 0) {
		period_end = period_start;
		linear_end = linear_start;
		angular_end = angular_start;

		period_start = m_time_buffer[buffer_index];
		linear_start = m_linear_buffer.col(buffer_index);
		angular_start = m_angular_buffer.col(buffer_index);
		buffer_index -= 1;

		// Now a few edge conditions to check
		// Sometimes we get velocity data for intervals of a few nanoseconds that get squished by the floating-point representation,
		// skip them to avoid zero-division issues
		if (period_start == period_end)
			continue;

		// If this time interval is entirely after the end_time, no need to dwell on it
		if (period_start > end_time)
			continue;
		
		// When we enter this loop, current_time is end_time or the last velocity data timestamp,
		// so either way, no need to worry about it as values will be integrated until current_time

		// Handle the start_times that are within this time period
		while (start_times[start_index] > period_start && start_index >= 0) {
			integrate_linear_velocity_reverse(translation.col(start_index), linear_start, linear_end, period_start, period_end, start_times[start_index], current_time);
			integrate_angular_velocity_reverse(rotation.col(start_index), angular_start, angular_end, period_start, period_end, start_times[start_index], current_time);
			current_time = start_times[start_index];
			start_index = next_start_time(translation, rotation, start_index);
		}

		if (start_index < 0)
			break;

		// Then the one that goes through to the previous time period
		integrate_linear_velocity_reverse(translation.col(start_index), linear_start, linear_end, period_start, period_end, period_start, current_time);
		integrate_angular_velocity_reverse(rotation.col(start_index), angular_start, angular_end, period_start, period_end, period_start, current_time);
		current_time = period_start;
	}

	// Now, we are just before the beginning of our velocity data
	// Some start_times might be before it, so we also need to extrapolate in that direction
	while (start_index >= 0) {
		extrapolate_linear_velocity_reverse(translation.col(start_index), linear_start, current_time - start_times[start_index]);
		integrate_angular_velocity_reverse_direct(rotation.col(start_index), angular_start, current_time - start_times[start_index]);
		current_time = start_times[start_index];
		start_index = next_start_time(translation, rotation, start_index);
	}

	// At this point, we have stored all of our transformations in the linear and angular transformation buffers
	// Now, let’s build transformation matrices out of them
	arma::fcube transforms(4, 4, num_starts);
	for (int i = 0; i < num_starts; i++) {
		// Formula for the rotation matrix from a quaternion from here :
		// https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
		// We normalize the quaternion at this step
		double rot_w = rotation(0, i), rot_x = rotation(1, i), rot_y = rotation(2, i), rot_z = rotation(3, i);
		double s = 1.0 / (rot_w*rot_w + rot_x*rot_x + rot_y*rot_y + rot_z*rot_z);
		
		// Rotation matrix from the resulting quaternion
		// We are in reverse, so this is the inverse (transposed) matrix
		transforms(0, 0, i) = static_cast<float>(1 - 2*s * (rot_y * rot_y + rot_z * rot_z));
		transforms(1, 0, i) = static_cast<float>(    2*s * (rot_x * rot_y - rot_z * rot_w));
		transforms(2, 0, i) = static_cast<float>(    2*s * (rot_x * rot_z + rot_y * rot_w));
		transforms(0, 1, i) = static_cast<float>(    2*s * (rot_x * rot_y + rot_z * rot_w));
		transforms(1, 1, i) = static_cast<float>(1 - 2*s * (rot_x * rot_x + rot_z * rot_z));
		transforms(2, 1, i) = static_cast<float>(    2*s * (rot_y * rot_z - rot_x * rot_w));
		transforms(0, 2, i) = static_cast<float>(    2*s * (rot_x * rot_z - rot_y * rot_w));
		transforms(1, 2, i) = static_cast<float>(    2*s * (rot_y * rot_z + rot_x * rot_w));
		transforms(2, 2, i) = static_cast<float>(1 - 2*s * (rot_x * rot_x + rot_y * rot_y));

		// Translation vector
		transforms(0, 3, i) = static_cast<float>(translation(0, i));
		transforms(1, 3, i) = static_cast<float>(translation(1, i));
		transforms(2, 3, i) = static_cast<float>(translation(2, i));

		// Affine part
		transforms(3, 0, i) = transforms(3, 1, i) = transforms(3, 2, i) = 0.0f;
		transforms(3, 3, i) = 1.0f;
	}

	return transforms;
}


/** Extrapolate the linear velocity with the given speed over some time interval
  * - translation  : arma::subview_col<double>[3]& : Translation vector to update
  * - linear_speed : arma::dvec[3]                 : Linear speed to apply
  * - timedelta    : double                        : Duration over which to apply the linear speed */
void TransformManager::extrapolate_linear_velocity_reverse(arma::subview_col<double>&& translation, arma::dvec const& linear_speed, double timedelta) {
	translation -= timedelta * linear_speed;
}

/** Integrate the linear velocity to compute the change in position between two time points
  * As we need everything from end to start, the result is reversed
  * - translation  : arma::subview_col<double>[3]& : Translation vector to update
  * - linear_start : arma::dvec[3]                 : Linear speed at the beginning of the time period
  * - linear_end   : arma::dvec[3]                 : Linear speed at the end of the time period
  * - period_start : double                        : Start of the velocity data time period
  * - period_end   : double                        : End of the velocity data time period
  * - start_time   : double                        : Time point at which to start the integration
  * - end_time     : double                        : Time point at which to end the integration */
void TransformManager::integrate_linear_velocity_reverse(arma::subview_col<double>&& translation, arma::dvec const& linear_start, arma::dvec const& linear_end, double period_start, double period_end, double start_time, double end_time) {
	// We have the linear speed at the beginning and at the end of the time period,
	// so we’ve got initial speed and constant acceleration
	// We can thus integrate this into a quadratic formula for the position
	// As always, we are in reverse so -=
	// It *is* (end_time + start_time) next to the acceleration, it’s because here (end_time² - start_time²) got factored into (end_time - start_time)(end_time + start_time)
	// and we need to subtract 2*period_start as it needs to be relative to period_start
	translation -= (end_time - start_time) * (linear_start + (end_time + start_time - 2*period_start) * (linear_end - linear_start) / (2*(period_end - period_start)));
}

// Now here’s the tricky part : integrating the linear velocity is trivial, but the angular velocity is on a whole other level
// From M. Boyle, The integration of angular velocity, 2017, the angular velocity can be integrated using the following differential equation :
// Ṙ(t) = 1/2 ω(t)R(t), where R(t) is the quaternion representing the rotation from the "world" (target) frame to the body frame, and
// ω(t) the angular velocity vector as a pure quaternion
// Now, this is a system of non-linear differential equations and I’m no quaternion expert,
// so let’s approximate it numerically using Taylor expansion and linear interpolation over small time steps
// instead of trying to actually solve this contraption

/** Integrate a constant angular velocity over some time interval
  * - rotation      : arma::subview_col<double>[4]& : Rotation quaternion to update
  * - angular_speed : arma::dvec[3]                 : Angular speed to apply
  * - timedelta     : double                        : Time interval to integrate */
void TransformManager::integrate_angular_velocity_reverse_direct(arma::subview_col<double>&& rotation, arma::dvec const& angular_speed, double timedelta) {
	// See this : https://math.stackexchange.com/questions/189185/quaternion-differentiation
	// Note that the angular speed is first stored in chronological order, but as we need the transform from the end to the start,
	// we need it in reverse, hence the minus sign instead of +
	// As long as the angular speed is not too high, the first-degree Taylor expansion seems satisfactory
	rotation[0] -= (timedelta / 2) * (-rotation[1] * angular_speed[0] - rotation[2] * angular_speed[1] - rotation[3] * angular_speed[2]);
	rotation[1] -= (timedelta / 2) * ( rotation[0] * angular_speed[0] + rotation[3] * angular_speed[1] - rotation[2] * angular_speed[2]);
	rotation[2] -= (timedelta / 2) * (-rotation[3] * angular_speed[0] + rotation[0] * angular_speed[1] + rotation[1] * angular_speed[2]);
	rotation[3] -= (timedelta / 2) * ( rotation[2] * angular_speed[0] - rotation[1] * angular_speed[1] + rotation[0] * angular_speed[2]);

	// Actually, normalizing the quaternion at intermediate steps doesn’t do us any significant good,
	// so let’s just leave it as is and normalize using the `s` factor when building the matrix
	// rotation = arma::normalise(rotation);
}

/** Integrate a linearly interpolated angular velocity over time
  * - rotation      : arma::subview_col<double>[4]& : Rotation quaternion to update
  * - angular_start : arma::dvec[3]                 : Angular velocity at the beginning of the period
  * - angular_end   : arma::dvec[3]                 : Angular velocity at the end of the time period
  * - period_start  : double      : Start of the time period
  * - period_end    : double      : End of the time period
  * - start_time    : double      : Time point at which to start the integration
  * - end_time      : double      : Time point at which to end the integration */
void TransformManager::integrate_angular_velocity_reverse(arma::subview_col<double>&& rotation, arma::dvec const& angular_start, arma::dvec const& angular_end, double period_start, double period_end, double start_time, double end_time) {
	double current_time = end_time;
	while (current_time > start_time) {
		// Get the time delta over which we’re going to integrate as constant angular velocity
		double timedelta = (current_time - start_time < m_sim_interval)? current_time - start_time : m_sim_interval;
		
		// Now get the linearly interpolated angular velocity
		arma::dvec current_angular_velocity = (current_time - period_start) * (angular_end - angular_start) / (period_end - period_start) + angular_start;
		
		// And integrate over this time step
		integrate_angular_velocity_reverse_direct(std::move(rotation), current_angular_velocity, timedelta);
		current_time -= timedelta;
	}
}

/** Convenience function to go to the next start time in the list :
	* Copy the current transform to the next index and return the next start index
	* - translation : arma::dmat[3, num_starts] : Array of translations
	* - rotation    : arma::dmat[4, num_starts] : Array of rotations
	* - start_index : int                       : Current start index. If it is 0, do not copy the transform and return a negative index
	* <-------------- int                       : Next start index */
int TransformManager::next_start_time(arma::dmat& translation, arma::dmat& rotation, int start_index) {
	// Copy if there is a next start time
	if (start_index > 0) {
		translation.col(start_index - 1) = translation.col(start_index);
		rotation.col(start_index - 1) = rotation.col(start_index);
	}
	return start_index - 1;
}