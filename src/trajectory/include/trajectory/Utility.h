#ifndef _TRAJECTORY_UTILITY_H
#define _TRAJECTORY_UTILITY_H

/** Various utility functions and macros */

#include <tuple>
#include <vector>
#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>

#include "config.h"
#include "fish2bird.h"

/** Take the square of a number */
#define sq(val) ((val)*(val))

/** Apply an affine transform to a batch of points, and return only as much rows there are in `points` */
inline arma::fmat affmul_direct(arma::fmat const& transform, arma::fmat const& points) {
	return arma::affmul(transform, points).eval().head_rows(points.n_rows);
}

inline arma::fmat affmul_direct(arma::fmat const& transform, arma::fmat&& points) {
	points = arma::affmul(transform, points).eval().head_rows(points.n_rows);
	return points;
}

/** Apply a 3D affine transform to a batch of 2D points, then return them projected back onto the x, y plane */
inline arma::fmat affmul_plane(arma::fmat const& transform, arma::fmat const& points) {
	return (transform.submat(0, 0, 1, 1) * points).eval().each_col() + transform.submat(0, 3, 1, 3);
}

/** Transform each vector by the corresponding affine matrix in `transforms`
 *  - transforms : arma::fcube[4, 4, N]          : Affine transform matrices
 *  - positions  : std::vector<arma::fvec[3]>[N] : Position vectors to transform
 *  <------------- arma::fmat[3, N]              : Transformed column vectors */
inline arma::fmat transform_positions(arma::fcube const& transforms, std::vector<arma::fvec> positions) {
	if (positions.empty())
		return arma::fmat();
	
	arma::fmat transformed(3, positions.size());
	for (int i = 0; i < positions.size(); i++)
		transformed.col(i) = affmul_direct(transforms.slice(i), positions[i]);
	return transformed;
}

/** Apply a 3D affine transform to 2D points 
  * - transform : arma::fmat[4, 4] : Affine transform matrix
  * - positions : arma::fmat[2, N] : Points to transform, Z is assumed to be 0
  * <------------ arma::fmat[3, N] : Transformed points */
inline arma::fmat transform_2d(arma::fmat const& transform, arma::fmat const& points) {
	arma::fmat points_3d = arma::join_cols(points, arma::zeros<arma::fmat>(1, points.n_cols));
	return affmul_direct(transform, points_3d);
}

/** Put the given vector on the columns of one matrix and the rows of the other, like the meshgrid() numpy function 
  * - vector : arma::fvec : Vector to make a grid from
  * <--------- arma::fmat : Column grid
  * <--------- arma::fmat : Row grid */
inline std::tuple<arma::fmat, arma::fmat> meshgrid(arma::fvec const& vector) {
	arma::fmat col_grid(vector.n_elem, vector.n_elem);
	arma::fmat row_grid(vector.n_elem, vector.n_elem);
	for (int i = 0; i < vector.n_elem; i++) {
		col_grid.col(i) = vector;
		row_grid.row(i) = vector.t();
	}
	
	return {col_grid, row_grid};
}

/** Convert a transform matrix to sXYZ convention Euler angles
  * Method taken from Matthew Brett’s transform3d Python package
  * - transform : arma::fmat[4, 4]     : Transform matrix
  * <------------ std::array<float, 3> : sXYZ Euler angles corresponding to the matrix */
inline std::array<float, 3> transform_to_sXYZ_euler(arma::fmat const& transform) {
	std::array<float, 3> angles;
	
	float cy = std::sqrt(sq(transform(0, 0)) + sq(transform(1, 0)));
	if (cy > 0) {
        angles[0] = std::atan2( transform(2, 1), transform(2, 2));
        angles[1] = std::atan2(-transform(2, 0), cy);
        angles[2] = std::atan2( transform(1, 0), transform(0, 0));
    } else {
        angles[0] = std::atan2(-transform(1, 2),  transform(1, 1));
        angles[1] = std::atan2(-transform(2, 0),  cy);
        angles[2] = 0.0f;
	}

	return angles;
}

/** Config wrappers for fish2bird */
inline arma::fmat birdeye_to_target_config(arma::fmat const& points) {
	return fish2bird::birdeye_to_target_2d(points, -config::birdeye::x_range, config::birdeye::x_range, config::birdeye::roi_y, config::birdeye::y_range, 
	                                       config::birdeye::birdeye_size, false, true);
}

inline arma::fmat birdeye_to_target_config(arma::fmat&& points) {
	return fish2bird::birdeye_to_target_2d(points, -config::birdeye::x_range, config::birdeye::x_range, config::birdeye::roi_y, config::birdeye::y_range, 
	                                       config::birdeye::birdeye_size, false, true);
}

inline arma::imat target_to_birdeye_config(arma::fmat const& points) {
	auto [target_points, scale_factor] = fish2bird::target_to_output(points, -config::birdeye::x_range, config::birdeye::x_range, config::birdeye::roi_y, config::birdeye::y_range, 
																	 config::birdeye::birdeye_size, false, true);
	return arma::conv_to<arma::imat>::from(target_points);
}

// Hash for combinations of indices
struct index_pair_hash {
	constexpr index_pair_hash (int max_index) : m_max_index(max_index) {}
    
	inline constexpr std::size_t operator()(std::pair<int,int> const& v) const {
        return v.first + 0x9e3779b9 + (v.second << 6) + (v.second >> 2);
    }

	int m_max_index;
};

// Conversion from armadillo curve to opencv polylines
namespace arma {
	template<> class conv_to<std::vector<cv::Point>> {
		public:
			inline static std::vector<cv::Point> from(umat const& points) {
				std::vector<cv::Point> cvpoints(points.n_cols);
				for (int point = 0; point < points.n_cols; point++)
					cvpoints[point] = cv::Point(points(0, point), points(1, point));
				return cvpoints;
			}

			inline static std::vector<cv::Point> from(fmat const& points) {
				std::vector<cv::Point> cvpoints(points.n_cols);
				for (int point = 0; point < points.n_cols; point++)
					cvpoints[point] = cv::Point(int(points(0, point)), int(points(1, point)));
				return cvpoints;
			}

			inline static std::vector<cv::Point> from(imat const& points) {
				std::vector<cv::Point> cvpoints(points.n_cols);
				for (int point = 0; point < points.n_cols; point++)
					cvpoints[point] = cv::Point(points(0, point), points(1, point));
				return cvpoints;
			}
	};
}

#endif