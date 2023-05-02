#ifndef _TRAJECTORY_DISCRETEGEOMETRY_H
#define _TRAJECTORY_DISCRETEGEOMETRY_H

#include <cmath>
#include <armadillo>

#include "trajectory/Utility.h"

/** Utility function, when we try to get the angle between two vectors using the dot product,
  * sometimes floating-points shenanigans make the cos value get a little bit outside [-1, 1],
  * so this puts it back into the right range and computes the arccos */
template<typename T> inline constexpr T acos_clamp(T val) {
	if      (val < -1) return M_PI;
	else if (val >  1) return 0;
	else               return std::acos(val);
}

/** Compute the angle of a vector relative to the main axis 
  * - vec : arma::fvec[D] : Vector to measure
  * <------ float         : Angle relative to the main axis in radians */
inline float vector_angle(arma::fvec const& vec) {
	return std::atan2(vec(1), vec(0));
}

/** Compute the relative angle between two vectors
  * - vec1 : arma::fvec[D] : First vector
  * - vec2 : arma::fvec[D] : Second vector
  * <------- float         : Relative angle in radians */
inline float vector_angle(arma::fvec const& vec1, arma::fvec const& vec2) {
	// That’s the usual dot product formula, arccos(v₁·v₂ / (||v₁|| × ||v₂||))
	return float(acos_clamp(arma::norm_dot(arma::conv_to<arma::dvec>::from(vec1), arma::conv_to<arma::dvec>::from(vec2))));
	// return acos_clamp(arma::norm_dot(vec1, vec2)));
}

/** Compute the orthogonal distance of a point to a line defined by an angle and a reference point
  * - input_point : Point to compute the distance of
  * - ref_point   : Any point on the line
  * - angle       : Angle the line makes with the trigonometric axis */
inline float distance_point_to_line(arma::fvec const& input_point, arma::fvec const& ref_point, float angle) {
	return std::abs(std::cos(angle) * (input_point(1) - ref_point(1)) - std::sin(angle) * (input_point(0) - ref_point(0)));
}

/** Compute the Savitzky-Golay filter coefficients for the given `window_size` and polynomial degree 2 or 3,
  * There are `window_size` coefficients */
inline arma::frowvec _savgol_coeffs_degree23(int window_size) {
	arma::frowvec coefficients(window_size);
	int i_first = (1 - window_size) / 2, i_last = (window_size - 1) / 2;
	for (int i = i_first; i <= i_last; i++)
		coefficients(i - i_first) = ((3.0f * sq(window_size) - 7.0f - 20.0f*sq(i)) / 4) / (window_size * (sq(window_size) - 4.0f) / 3.0f);
	return coefficients;
}

/** Compute the physical length of a discrete curve
  * - curve : arma::fmat[2, N] : Input curve
  * <-------- float            : Length of the curve */
inline float curve_length(arma::fmat const& curve) {
	float result = 0;
	for (int i = 0; i < curve.n_cols - 1; i++)
		result += arma::norm(curve.col(i + 1) - curve.col(i));
	return result;
}

/** Compute the norm of each column vector
  * - vectors : arma::fmat    : Vectors to take the norm of
  * <---------- arma::frowvec : Norm of each vector */
inline arma::frowvec column_norm(arma::fmat const& vectors) {
	return arma::sqrt(arma::sum(arma::square(vectors), 0));
}

/** Get the orthogonal vectors of the given column vectors 
  * - vectors : arma::fmat : Input vectors
  * <---------- arma::fmat : Orthogonal vectors */
inline arma::fmat column_orthogonal(arma::fmat const& vectors) {
	arma::fmat orthogonal(vectors);
	orthogonal.swap_rows(0, 1);
	orthogonal.row(0) *= -1;
	return orthogonal;
}

inline arma::fmat column_orthogonal(arma::fmat&& vectors) {
	vectors.swap_rows(0, 1);
	vectors.row(0) *= -1;
	return vectors;
}

/** Check whether the segments AB and CD intersect with each other
  * - A : arma::fvec[2] : First point of the first segment
  * - B : arma::fvec[2] : Second point of the first segment
  * - C : arma::fvec[2] : First point of the second segment
  * - D : arma::fvec[2] : Second point of the second segment
  * <---- bool          : Whether the segments AB and CD intersect */
inline bool segments_intersect(arma::fvec const& A, arma::fvec const& B, arma::fvec const& C, arma::fvec const& D) {
	float determinant_A = (C(0)-A(0)) * (D(1)-A(1)) - (D(0)-A(0)) * (C(1)-A(1));
	float determinant_B = (C(0)-B(0)) * (D(1)-B(1)) - (D(0)-B(0)) * (C(1)-B(1));
	float determinant_C = (A(0)-C(0)) * (B(1)-C(1)) - (B(0)-C(0)) * (A(1)-C(1));
	float determinant_D = (A(0)-D(0)) * (B(1)-D(1)) - (B(0)-D(0)) * (A(1)-D(1));

	return (((determinant_A > 0 && determinant_B < 0) || (determinant_A < 0 && determinant_B > 0)) &&
	        ((determinant_C > 0 && determinant_D < 0) || (determinant_C < 0 && determinant_D > 0)));
}

/** Get the indices of the points to keep to remove the self-intersections of a curve
  * - curve : arma::fmat : Tangled curve
  * <-------- arma::uvec : Indices to keep */
inline arma::uvec find_self_intersects(arma::fmat const& curve) {
	// Check for intersection between each pair of segments
	// When two segments intersect, cut out everything in-between such that the points
	// before the first segment and after the second segment are linked, removing the intersecting part
	arma::uvec mask(curve.n_cols, arma::fill::ones);

	for (int i = 0; i < curve.n_cols - 1; i++)
		for (int j = i + 1; j < curve.n_cols - 1; j++)
			if (segments_intersect(curve.col(i), curve.col(i + 1), curve.col(j), curve.col(j + 1)))
				mask.subvec(i + 1, j).fill(false);
	
	return arma::find(mask);
}

/** Compute the mean curvature of the given curve, in rad/unit
  * - curve : arma::fmat[2, N] : Curve to process
  * <-------- float            : Mean of the local curvatures in rad/length unit */
inline float mean_curvature(arma::fmat const& curve) {
	arma::fmat vectors = arma::diff(curve, 1, 1);
	arma::frowvec vector_norms = column_norm(vectors);
	
	// Our local curvature is the angle between the incoming and outgoing vectors, divided by the mean local segment length
	float curvature_sum = 0;
	for (int i = 0; i < vectors.n_cols - 1; i++)
		curvature_sum += 2 * vector_angle(vectors.col(i), vectors.col(i + 1)) / (vector_norms(i) + vector_norms(i + 1));
	
	return curvature_sum / (vectors.n_cols - 1);
}


#endif