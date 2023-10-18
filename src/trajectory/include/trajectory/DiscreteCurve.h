#ifndef _TRAJECTORY_DISCRETECURVE_H
#define _TRAJECTORY_DISCRETECURVE_H

#include <tuple>
#include <cmath>
#include <cassert>
#include <unordered_map>
#include <armadillo>

#include "ros/ros.h"
#include "trajectory/DiscreteGeometry.h"


struct DiscreteCurve {
	arma::fmat curve;
	arma::frowvec scores;
	ros::Time timestamp;

	inline DiscreteCurve() = default;
	inline DiscreteCurve(arma::fmat&& curve_) : curve(curve_) {}
	inline DiscreteCurve(arma::fmat const& curve_) : curve(curve_) {}
	inline DiscreteCurve(arma::fmat&& curve_, ros::Time timestamp_) : curve(curve_), timestamp(timestamp_) {}
	inline DiscreteCurve(arma::fmat const& curve_, ros::Time timestamp_) : curve(curve_), timestamp(timestamp_) {}
	inline DiscreteCurve(arma::fmat&& curve_, arma::frowvec&& scores_) : curve(curve_), scores(scores_) {}
	inline DiscreteCurve(arma::fmat const& curve_, arma::frowvec const& scores_) : curve(curve_), scores(scores_) {}
	inline DiscreteCurve(arma::fmat&& curve_, arma::frowvec&& scores_, ros::Time timestamp_) : curve(curve_), scores(scores_), timestamp(timestamp_) {}
	inline DiscreteCurve(arma::fmat const& curve_, arma::frowvec const& scores_, ros::Time timestamp_) : curve(curve_), scores(scores_), timestamp(timestamp_) {}

	inline bool is_valid() const {
		return !curve.is_empty();
	}

	inline bool has_scores() const {
		return !scores.is_empty();
	}

	inline int size() const {
		return curve.n_cols;
	}

	/** Compute the (physical) length of the curve, as the sum of the length of each segment */
	inline float length() const {
		return curve_length(curve);
	}

	/** Compute the gradient of the curve
	  * <---- arma::fmat[2, N] : Gradient vectors along the curve */
	inline arma::fmat gradient() const {
		const arma::frowvec filter = {0.5, 0, -0.5};
		arma::fmat result(2, curve.n_cols);
		result.row(0) = arma::conv(curve.row(0), filter, "same");
		result.row(1) = arma::conv(curve.row(1), filter, "same");
		result.col(0) = curve.col(1) - curve.col(0);
		result.col(curve.n_cols - 1) = curve.col(curve.n_cols - 1) - curve.col(curve.n_cols - 2);
		return result;
	}

	/** Get the actual coordinates of a point given as [index, segment_part] 
	  * - index            : int           : Index of the segment on the curve
	  * - segment_part = 0 : float         : Portion of the outgoing vector to add to get to the actual point 
	  * <--------------- arma::fvec[2] : Coordinates of the point */
	inline arma::fvec get_point(int index, float segment_part=0) const {
		// Special case for the last point
		if (index == size() - 1 && segment_part == 0)
			return curve.col(index);
		
		assert(index >= 0 && index < size() - 1 && segment_part >= 0);
		return (1 - segment_part) * curve.col(index) + segment_part * curve.col(index + 1);
	}

	/** Get the score associated with a point, even in-between samples
	  * - index            : int   : Index of the segment on the curve
	  * - segment_part = 0 : float : Portion of the outgoing vector to add to get to the actual point
	  * <--------------- float : Score associated to the point */
	inline float get_score(int index, float segment_part=0) const {
		if (index == size() - 1 && segment_part == 0)
			return scores(index);
		
		assert(index >= 0 && index < size() - 1 && segment_part >= 0);
		return (1 - segment_part) * scores(index) + segment_part * scores(index + 1);
	}

	/** Find a point along the curve that is `target_distance` further than the given point along the curve
	  * - index           : int   : Index of the initial point on the curve
	  * - segment_part    : float : Portion of the outgoing vector of the given index on the curve to add to get the initial point
	  * - target_distance : float : Distance from the initial point to the result
	  * <------------------ int   : Index of the sample immediately before the resulting point
	  *                             If the point is beyond the curve, on any end, that index is negative
	  * <------------------ float : Portion of the outgoing vector of the resulting index on the curve to add to get the resulting point */
	inline std::tuple<int, float> next_point(int index, float segment_part, float target_distance) const {
		// Loop on the curve samples starting from the initial projection until the cumulative distance adds up to `target_distance` or it overshoots the curve
		float distance = 0;
		while (index < size() - 1) {
			arma::fvec outgoing_vector = curve.col(index + 1) - curve.col(index);

			// When the segment_part is non-null, take the part of the vector that remains *after* the segment_part
			// To start at the actual initial point, including its vector part
			arma::fvec remaining_vector = (1 - segment_part) * outgoing_vector;
			float remaining_length = arma::norm(remaining_vector);

			// This entire vector would make us overshoot the target : the resulting point is between this sample and the next
			// We just need to compute the necessary proportion of the vector
			if (distance + remaining_length > target_distance) {
				float outgoing_vector_length = arma::norm(outgoing_vector);
				segment_part += (target_distance - distance) / outgoing_vector_length;
				return {index, segment_part};
			}

			// Otherwise, just add this distance and continue on to the next sample
			else {
				index += 1;
				segment_part = 0;
				distance += remaining_length;
			}
		}

		// Overshot the curve, fail
		return {-1, 0};
	}
	
	/** Resample a curve at regular intervals
	  * There is no guarantee on the actual number of points in the resampled curve
	  * Actually, the end of the curve could be cut off if the last segment is too short
	  * - step : float : Length of each segment of the resampled curve */
	inline void resample(float step) {
		int num_samples = std::ceil(length() / step);
		if (num_samples == 0) num_samples = 1;

		arma::fmat resampled(2, num_samples);
		resampled.col(0) = curve.col(0);

		int index = 0;
		float segment_part = 0;
		for (int i = 1; i < num_samples; i++) {
			std::tie(index, segment_part) = next_point(index, segment_part, step);
			assert(index >= 0);  // Passed the end of the curve, this should not happen
			resampled.col(i) = get_point(index, segment_part);
		}

		curve = std::move(resampled);
	}

	/** Apply a Savitzky-Golay filter on the given curve samples
	  * - window_size   : int          : Filter window size, must be odd and greater than `savgol_degree` */
	inline void savgol_filter(int window_size) {
		static_assert(config::lines::savgol_degree == 2 || config::lines::savgol_degree == 3);
		
		// Compute the best window size for this specific curve
		// It must be odd and it must be lower or equal to the curve size
		if (window_size <= config::lines::savgol_degree)
			window_size = config::lines::savgol_degree + (1 - config::lines::savgol_degree % 2);
		else if (window_size < size())
			window_size -= (1 - window_size % 2);
		else
			window_size = size() - (1 - size() % 2);
		
		// Compute the coefficients, or get them from the cache
		arma::frowvec coefficients = _savgol_coeffs_degree23(window_size);
		
		// Apply the convolution
		// To avoid squishing the extreme points, extrapolate the curve using its extreme vectors
		const int i_first = (1 - window_size) / 2;
		arma::fvec first_vector = curve.col(0) - curve.col(1);
		arma::fvec last_vector = curve.col(curve.n_cols - 1) - curve.col(curve.n_cols - 2);
		arma::fmat result(2, curve.n_cols, arma::fill::zeros);
		for (int x = 0; x < curve.n_cols; x++) {
			for (int i = 0; i < window_size; i++) {
				arma::fvec conv_vector;
				int conv_index = x + i + i_first;

				// Index < 0 ⟶ before the first sample, use the first vector
				if (conv_index < 0)
					conv_vector = curve.col(0) - first_vector * conv_index;
				
				// Index ≥ curve.n_cols ⟶ beyond the last sample, extend with the last vector
				else if (conv_index >= curve.n_cols)
					conv_vector = curve.col(curve.n_cols - 1) + last_vector * (conv_index - (curve.n_cols - 1));
				else
					conv_vector = curve.col(conv_index);
				
				result.col(x) += conv_vector * coefficients(i);
			}
		}

		curve = std::move(result);
	}

	/** Return the closest point on the curve to the given point, including in-between curve samples
	  * - point         : arma::fvec[2] : Point to project
	  * - extend        : bool          : If true, when the projection is before the first sample of the curve, extend it
	  * - min_index = 0 : int           : Minimum index to project to
	  * <---------------- int           : Index of the closest sample to the projection in the curve
	                                      If the projection is beyond the last sample of the curve, set to -1
										  If the projection is before the first sample of the curve and `extend` is false, set to -1
	  * <---------------- float         : Portion of the segment that follows the selected sample
	                                      If the projection is before the first segment of the curve and `extend` is true, return a null index and a negative segment factor */
	inline std::tuple<int, float> project_point(arma::fvec const& point, bool extend, int min_index=0) const {
		int best_index = -1;
		bool best_orthogonal = false;
		float best_part = 0, best_distance = arma::datum::inf;
		
		for (int i = 0; i < curve.n_cols - 1; i++) {
			arma::fvec current_point = curve.col(i);
			arma::fvec current_vector = curve.col(i + 1) - curve.col(i);

			// Find the proportion of the vector at which the point orthogonally projects
			// If it is within [0, 1[, then it can be better projected at a point on that vector
			// Otherwise we continue with the curve sample alone
			float part = arma::dot(point - current_point, current_vector) / arma::dot(current_vector, current_vector);
			if ((part >= 0 || (i == 0 && extend)) && part < 1)
				current_point += current_vector*part;
			else
				part = 0;
			
			// Take the square distance from the point to project to the current point or the projection onto the current vector
			// The overall goal is to minimize this distance
			float distance = arma::norm(point - current_point);
			if (distance < best_distance) {
				best_index = i;
				best_part = part;
				best_distance = distance;

				// If the segment part is 0, that means the closest thing to the projected point is the actual sample
				// So it’s difficult to tell whether it’s a real orthogonal projection
				best_orthogonal = part > 0;
			}
		}

		// Also check the distance to the last curve sample
		// If it is closest to the point to project, then the point is beyond the curve and we fail
		// However, this might be falsified by high curvatures near the end, so if the best point found previously
		// was definitely orthogonal, keep it instead
		float last_distance = arma::norm(point - curve.col(curve.n_cols - 1));
		if (last_distance < best_distance && !best_orthogonal)
			return {-1, -1};
		
		// If the point is before the first sample and no extension was required, fail
		if (best_index < 0 || (best_index == 0 && best_part <= 0 && !extend))
			return {-1, -1};
		else
			return {best_index, best_part};
	}

	/** Apply a "dilation" operation to the curve, that is, offset it regularly regardless of the curve orientation :
	  * all points of the dilated curve are approximately at the same distance of their corresponding original point
	  * - dilation  : float : Distance of the result to the original curve
	  * - direction : int   : 1 or -1, to choose the offset direction */
	inline void dilate(float dilation, int direction) {
		// First step : offset the curve by `dilation` along its orthogonal vectors
		arma::fmat orthogonal_vectors = column_orthogonal(gradient());
		arma::frowvec gradient_norm = column_norm(orthogonal_vectors);
		arma::fmat dilated = curve + dilation * (orthogonal_vectors.each_row() / gradient_norm) * direction;

		// Second step : Remove self-intersections
		arma::uvec filter = find_self_intersects(dilated);
		curve = dilated.cols(filter);
		if (!scores.is_empty())
			scores = scores.cols(filter);
	}

	/** Cut out the points that make too sharp angles in the curve
	   - max_angle : double       : Cut angles that are sharper than this (so, lower angle value), in radians */
	inline void cut_angles(float max_angle) {
		arma::fmat vectors = arma::diff(curve, 1, 1);
		arma::frowvec angles(curve.n_cols - 2);
		for (int i = 0; i < vectors.n_cols - 1; i++)
			angles(i) = vector_angle(vectors.col(i), vectors.col(i + 1));
		
		arma::uvec valid_indices = arma::find(angles <= max_angle) + 1;
		curve = curve.cols(valid_indices);
		if (!scores.is_empty())
			scores = scores.cols(valid_indices);
	}
};

/** Transform each curve by the corresponding affine matrix in `transforms`
 *  - transforms : arma::fcube[4, 4, N]          : Affine transform matrices
 *  - curves     : std::vector<DiscreteCurve>[N] : Curves to transform
 *  <------------- std::vector<DiscreteCurve>[N] : Transformed column vectors */
inline std::vector<DiscreteCurve> transform_positions(arma::fcube const& transforms, std::vector<DiscreteCurve> const& curves) {
	std::vector<DiscreteCurve> transformed;
	if (curves.empty())
		return transformed;
	
	for (int i = 0; i < curves.size(); i++)
		transformed.emplace_back(affmul_plane(transforms.slice(i), curves[i].curve), curves[i].scores, curves[i].timestamp);
	return transformed;
}

/** Transform the curve by the corresponding affine matrix in `transform`
 *  - transforms : arma::fmat[4, 4] : Affine transform matrix
 *  - curves     : DiscreteCurve    : Curve to transform
 *  <------------- DiscreteCurve    : Transformed curve */
inline DiscreteCurve transform_positions(arma::fmat const& transform, DiscreteCurve const& curve) {
	return DiscreteCurve(affmul_plane(transform, curve.curve), curve.scores, curve.timestamp);
}

float mean_parallel_distance(DiscreteCurve const& curve1, DiscreteCurve const& curve2);
DiscreteCurve compile_line(std::vector<DiscreteCurve> const& lines, float min_score, arma::fvec const& start_point, float trajectory_step);


#endif