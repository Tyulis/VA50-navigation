#ifndef _TRANSFORMTRACK_TRANSFORMMANAGER_H
#define _TRANSFORMTRACK_TRANSFORMMANAGER_H

#include <vector>
#include <cassert>

#include <armadillo>

class TransformManager {
	public:
		TransformManager();
		TransformManager(double sim_interval);
	
		void add_velocity(double timestamp, arma::dvec linear, arma::dvec angular);
		void drop_velocity(double end_time);
		arma::fcube get_map_transforms(arma::dvec const& start_times, double end_time);
	
	private:
		void extrapolate_linear_velocity_reverse(arma::subview_col<double>&& translation, arma::dvec const& linear_speed, double timedelta);
		void integrate_linear_velocity_reverse(arma::subview_col<double>&& translation, arma::dvec const& linear_start, arma::dvec const& linear_end, double period_start, double period_end, double start_time, double end_time);
		void integrate_angular_velocity_reverse_direct(arma::subview_col<double>&& rotation, arma::dvec const& angular_speed, double timedelta);
		void integrate_angular_velocity_reverse(arma::subview_col<double>&& rotation, arma::dvec const& angular_start, arma::dvec const& angular_end, double period_start, double period_end, double start_time, double end_time);
		int next_start_time(arma::dmat& translation, arma::dmat& rotation, int start_index);

		double m_sim_interval;              // "Simulation interval", duration of the integration intervals for the angular velocity
		std::vector<double> m_time_buffer;  // Timestamps in seconds (with decimal part)
		arma::dmat m_linear_buffer;         // Linear speeds along each axis at each timestamp
		arma::dmat m_angular_buffer;        // Angular velocity vector at each timestep, as pure quaternions
};


#endif