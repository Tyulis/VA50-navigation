#include <stdexcept>

#include "trajectory/Statistics.h"
#include "trajectory/DiscreteCurve.h"

#define CLAMP(x, min, max) (((x) > (max))? (max) : (((x) < (min))? (min) : (x)))


/** Compile a single curve and its confidence scores from a group of discrete curves
  * - lines           : std::vector<DiscreteCurve> : Curves to compile
  * - min_score       : float                      : Points below this score will not be considered
  * - start_point     : arma::fvec[2]              : Initial point of the compiled curve
  * - trajectory_step : float                      : Interval at which the points are taken
  * <------------------ DiscreteCurve              : Compiled curve */
DiscreteCurve compile_line(std::vector<DiscreteCurve> const& lines, float min_score, arma::fvec const& start_point, float trajectory_step) {
	// Nothing to compile, so compile nothing
	if (lines.size() == 0)
		return DiscreteCurve();
	
	// Get an upper limit on where the compiled trajectory may go
	float max_y = 0;
	for (auto it = lines.begin(); it != lines.end(); it++) {
		float curve_max_y = arma::max(it->curve.row(1));
		if (curve_max_y > max_y)
			max_y = curve_max_y;
	}
	
	// Each point of the final curve is the weighted mean of the intersection of
	// each curve with circle arcs set at intervals of `trajectory_step`
	arma::uvec last_indices(lines.size(), arma::fill::zeros);
	std::vector<float> result_angles_vec = {std::atan2(start_point(1), start_point(0))};
	std::vector<float> result_scores_vec = {1};
	std::vector<float> radii_vec = {arma::norm(start_point)};
	const arma::fvec quantile_def = {0.25, 0.5, 0.75};

	for (float radius = radii_vec[0] + trajectory_step; radius < max_y; radius += trajectory_step) {
		// Get the angle at which each curve intersects the arc
		std::vector<float> intersect_angles_vec;
		std::vector<float> intersect_scores_vec;
		for (int i = 0; i < lines.size(); i++) {
			DiscreteCurve line = lines[i];

			// Advance along the curve from the last found index until it gets to the required distance
			for (int p = last_indices(i); p < line.size(); p++) {
				float point_distance = arma::norm(line.curve.col(p));
				if (point_distance > radius) {
					// Before the first point on the curve : nothing here
					if (p == 0)
						break;
					
					// Compute the coefficient to put on the segment vector to get to the intersection from the previous point
					// x = x₀ + kVx, y = y₀ + kVy, x² + y² = r² as we start from the origin
					// This makes us solve (Vx² + Vy²)k² + 2(x₀Vx + y₀Vy)k + (x₀² + y₀²) - r² = 0 for k
					arma::fvec base_point = line.curve.col(p - 1);
					arma::fvec vector = line.curve.col(p) - base_point;
					
					float a = sq(vector(0)) + sq(vector(1));
					float b = 2 * (vector(0)*base_point(0) + vector(1)*base_point(1));
					float c = sq(base_point(0)) + sq(base_point(1)) - sq(radius);
					float discriminant = sq(b) - 4*a*c;

					// The previous segment doesn’t intersect the circle : ??
					if (discriminant < 0)
						throw std::logic_error("Incoherent circle intersections");
					
					float segment_part;
					float k1 = (-b - std::sqrt(discriminant)) / (2 * a);
					float k2 = (-b + std::sqrt(discriminant)) / (2 * a);
					
					// Take the solution that’s actually on the segment, so with k ∈ [0, 1[
					// Sometimes the floating-point stability screws us over, hence the tiny bit allowed around that range
					if (k1 >= -0.00001 && k1 < 1.00001)
						segment_part = CLAMP(k1, 0, 1);
					else if (k2 >= -0.00001 && k2 < 1.00001)
						segment_part = CLAMP(k2, 0, 1);

					// None of the solutions are on the segment : ??
					else throw std::logic_error("No solutions on the segment");

					arma::fvec point = line.get_point(p - 1, segment_part);
					float score = line.get_score(p - 1, segment_part);
					if (score > min_score) {
						intersect_angles_vec.push_back(std::atan2(point(1), point(0)));
						intersect_scores_vec.push_back(score);
						last_indices(i) = p - 1;
					}
					break;
				}
			}
		}

		// No points interesting enough
		// If there are no points in the result list yet, the closest points might just be too far extrapolated, 
		// so skip to the next step to see if it is better, and cut if it has gone the whole way without finding anything
		// If there are already result points, stop here as it was most likely the last interesting position
		if (intersect_angles_vec.size() == 0) {
			if (result_angles_vec.size() == 1 && radius)
				continue;
			else
				break;
		}

		arma::frowvec intersect_angles = arma::conv_to<arma::frowvec>::from(intersect_angles_vec);
		arma::frowvec intersect_scores = arma::conv_to<arma::frowvec>::from(intersect_scores_vec);
		float centroid;

		// One point, the centroid is easy enough
		if (intersect_angles.n_elem == 1) {
			result_angles_vec.push_back(intersect_angles(0));
			result_scores_vec.push_back(intersect_scores(0));
		} else {
			// Perform an interquartile range filtering to discard blatantly wrong points
			arma::frowvec quantiles = arma::quantile(intersect_angles, quantile_def);

			// If by chance all curves get approximately the same angle at the same time, the quantiles are all the same
			// In that case, there’s actually a single value for all of them
			if (arma::all(quantiles - quantiles(0) < 0.00001)) {
				result_angles_vec.push_back(arma::mean(intersect_angles));
				result_scores_vec.push_back(confidence_combination(intersect_scores));
			} else {
				arma::uvec iqr_filter = arma::find((quantiles(1) - 1.5 * (quantiles(2) - quantiles(0)) >= intersect_angles) || (intersect_angles >= quantiles(1) + 1.5 * (quantiles(2) - quantiles(0))));
				intersect_angles.shed_cols(iqr_filter);
				intersect_scores.shed_cols(iqr_filter);
				result_angles_vec.push_back(arma::mean(intersect_angles));
				result_scores_vec.push_back(confidence_combination(intersect_scores));
			}
		}

		radii_vec.push_back(radius);
	}

	arma::frowvec result_angles = arma::conv_to<arma::frowvec>::from(result_angles_vec);
	arma::frowvec result_scores = arma::conv_to<arma::frowvec>::from(result_scores_vec);
	arma::frowvec radii = arma::conv_to<arma::frowvec>::from(radii_vec);
	
	arma::fmat result_points(2, result_angles.n_elem);
	result_points.row(0) = radii % arma::cos(result_angles);
	result_points.row(1) = radii % arma::sin(result_angles);

	return DiscreteCurve(std::move(result_points), std::move(result_scores));
}

float mean_parallel_distance(DiscreteCurve const& curve1, DiscreteCurve const& curve2) {
	DiscreteCurve longest_line;
	DiscreteCurve shortest_line;
	if (curve1.length() > curve2.length()) {
		longest_line = curve1;
		shortest_line = curve2;
	} else {
		longest_line = curve2;
		shortest_line = curve1;
	}

	// Now compute the average orthogonal distance and the average angle difference
	float paralleldiff = 0.0f;
	int valid_points = 0;
	for (int p = 0; p < longest_line.size() - 1; p++) {
		auto [index, segment_part] = shortest_line.project_point(longest_line.curve.col(p), false);
		if (index < 0)
			continue;

		paralleldiff += arma::norm(longest_line.get_point(p) - shortest_line.get_point(index, segment_part));
		valid_points += 1;
	}

	if (valid_points == 0)
		return INFINITY;
	else
		return paralleldiff / valid_points;
}