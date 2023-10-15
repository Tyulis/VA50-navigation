#include <set>
#include <array>

#include "trajectory/Utility.h"
#include "trajectory/LineDetection.h"
#include "trajectory/DiscreteGeometry.h"


/** Count the 8-neighbors of a pixel in `remainder`
  * - remainder      : cv::Mat[y, x, CV_8U] : Binary image in which to count the neighbors
  * - base_y, base_x : int                  : Pixel to count the neighbors of
  * <----------------- int                  : Number of 8-neighbors of that pixel */
static int count_neighbors(cv::Mat const& remainder, int base_y, int base_x) {
	int count = 0;
	for (int y = base_y - 1; y <= base_y + 1; y++) {
		if (y < 0 || y >= remainder.rows)
			continue;
		
		for (int x = base_x - 1; x <= base_x + 1; x++) {
			if ((x == base_x && y == base_y) || x < 0 || x >= remainder.cols)
				continue;
			
			if (remainder.at<uint8_t>(y, x) > 0)
				count += 1;
		}
	}

	return count;
}


/** Extract a single curve in the image, starting from point (base_y, base_x)
  * - remainder      : cv::Mat[y, x, CV_8U]   : Binary values of the image, pixels at 0 are ignored
  * - branch         : std::vector<cv::Point> : Branch to fill with discrete curve points, in order
  * - base_y, base_x : int                    : Position at which to start the extraction
  * - init_y, init_x : int                    : Position to consider as the previous point on the curve. Use only internal, set to -1 for unused. */
void extract_branch(cv::Mat& remainder, std::vector<cv::Point>& branch, cv::Point base_point, cv::Point init_point) {
	cv::Point last_point;
	cv::Point neighbors[8];
	std::vector<cv::Point> subbranch;
	
	// Return when there are no neighbors remaining
	while (true) {
		// Add the current point on the branch and remove it from the remainder
		branch.push_back(base_point);
		remainder.at<uint8_t>(base_point) = 0;

		// Find the 8-neighborhood of the current base point
		int neighbor_index = 0;
		for (int y = base_point.y - 1; y <= base_point.y + 1; y++) {
			if (y < 0 || y >= remainder.rows)
				continue;
			
			for (int x = base_point.x - 1; x <= base_point.x + 1; x++) {
				if ((x == base_point.x && y == base_point.y) || x < 0 || x >= remainder.cols)
					continue;
				if (remainder.at<uint8_t>(y, x) > 0) {
					neighbors[neighbor_index] = cv::Point(x, y);
					neighbor_index += 1;
				}
			}
		}

		// No neighbors anymore : quit
		if (neighbor_index == 0)
			return;
		
		// One neighbor : just add it and go forth with it
		else if (neighbor_index == 1)
			base_point = neighbors[0];

		// Two or more neighbors : some selection ought to take place
		else if (neighbor_index >= 2) {
			// First point on the branch and no init position, this is the initial point on the branch and goes multiple ways
			// Select the two farthest apart neighbors and add them at both ends of the branch
			if (branch.size() == 1 && init_point.y < 0 && init_point.x < 0) {
				// Select the neighbors that are farthest apart
				int best_neighbors[2];
				int max_sqdistance = 0;
				for (int i = 0; i < neighbor_index; i++) {
					for (int j = i + 1; j < neighbor_index; j++) {
						cv::Point diff_vector = neighbors[i] - neighbors[j];
						int sqdistance = diff_vector.dot(diff_vector);
						if (sqdistance > max_sqdistance) {
							max_sqdistance = sqdistance;
							best_neighbors[0] = i;
							best_neighbors[1] = j;
						}
					}
				}

				// Remove the other neighbors, otherwise they could parasite the following loops
				// Those are just line overthickness, if it is actually a branching curve, it will just
				// clear the first pixel and be caught back later
				for (int i = 0; i < neighbor_index; i++)
					if (i != best_neighbors[0] && i != best_neighbors[1])
						remainder.at<uint8_t>(neighbors[i]) = 0;
				
				// Max square distance = 1 -> the neighbors farthest apart are adjacent pixels,
				// so it’s actually one side of an overthick diagonal branch,
				// go with any of those two neighbors and ditch the other
				if (max_sqdistance == 1) {
					remainder.at<uint8_t>(neighbors[best_neighbors[1]]) = 0;
					base_point = neighbors[best_neighbors[0]];
				}

				// Otherwise, we need to go both ways. To do that, call _extract_branch recursively with a temporary array,
				// and an initial position to make sure it goes in the right direction (see the next part of the function)
				else {
					subbranch.clear();
					extract_branch(remainder, subbranch, neighbors[best_neighbors[0]], base_point);

					// Then add it in reverse at the beginning
					std::vector<cv::Point> result;
					result.reserve(subbranch.size() + branch.size());
					for (auto rit = subbranch.rbegin(); rit != subbranch.rend(); rit++)
						result.push_back(*rit);
					result.insert(result.end(), branch.begin(), branch.end());
					branch = result;

					base_point = neighbors[best_neighbors[1]];
				}
			}

			// There is a previous position, so we just need to select the one neighbor that is farthest from it
			else {
				// First point and initial position, we are in a recursive call and there are multiple neighbors,
				// select the neighbor farthest from the initial point of the parent branch to make sure it goes in the right direction
				if (branch.size() == 1 && (init_point.y >= 0 and init_point.x >= 0))
					last_point = init_point;
				
				// Just randomly found multiple neighbors on the curve, the previous position is just the previous point
				else
					last_point = branch.back();
				
				// Maximize the distance to the previous position and find the farthest neighbor
				int max_sqdistance = 0;
				int best_neighbor = -1;
				for (int i = 0; i < neighbor_index; i++) {
					cv::Point diff_vector = neighbors[i] - last_point;
					int sqdistance = diff_vector.dot(diff_vector);
					if (sqdistance > max_sqdistance) {
						max_sqdistance = sqdistance;
						best_neighbor = i;
					}
				}

				// Delete the other neighbors completely, otherwise they could parasite the following loops
				// Those are just line overthickness, if it is actually a branching curve, it will just
				// clear the first pixel and be caught back later
				for (int i = 0; i < neighbor_index; i++)
					if (i != best_neighbor)
						remainder.at<uint8_t>(neighbors[i]) = 0;
				
				// And go to the next one
				base_point = neighbors[best_neighbor];
			}
		}
	}
}


/** Extract all "branches", continuous, 1-pixel wide discrete curves in a binary image
  * The points within a branch are in order, with no discontinuity
  * - image    : cv::Mat[y, x, CV_8U]       : Binary image in which to extract curves
  * <----------- std::vector<DiscreteCurve> : List of discrete curves in (x, y)ᵀ pixel coordinates */
std::vector<DiscreteCurve> extract_branches(cv::Mat const& image) {
	std::vector<DiscreteCurve> branches;
	
	// We need a state mask to keep track of which pixels remain to be processed
	// `remainder` is our state mask, 0 = to ignore (originally black pixel, or already done),
	// 1 = to be processed
	cv::Mat remainder = cv::Mat::zeros(image.rows, image.cols, CV_8U);
	for (int y = 0; y < image.rows; y++)
		for (int x = 0; x < image.cols; x++)
			if (image.at<uint8_t>(y, x) > 0)
				remainder.at<uint8_t>(y, x) = 1;
			else
				remainder.at<uint8_t>(y, x) = 0;
	
	// Now loop over the remainder, and extract a branch whenever we find an interesting point
	// All curve points should be processed in one go
	std::vector<cv::Point> branch_build;
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			// Nothing to see here
			if (remainder.at<uint8_t>(y, x) == 0)
				continue;
			
			// Check how many 8-neighbors this point has to know what to do
			int num_neighbors = count_neighbors(remainder, y, x);

			// Isolated point -> just discard it
			if (num_neighbors == 0) {
				remainder.at<uint8_t>(y, x) = 0;
			} else if (num_neighbors >= 1) {
				branch_build.clear();
				extract_branch(remainder, branch_build, cv::Point(x, y), cv::Point(-1, -1));
				if (branch_build.size() > 1) {
					arma::fmat branch(2, branch_build.size());
					for (int col = 0; col < branch_build.size(); col++) {
						branch(0, col) = branch_build[col].x;
						branch(1, col) = branch_build[col].y;
					}
					branches.emplace_back(std::move(branch));
				}
			}
		}
	}

	return branches;
}


/** Take a list of discrete curves, smooth them, split them at angles and merge those that would make continuous curves,
  * such that the result is a set of smooth discrete curves
  * - branches     : std::vector<DiscreteCurve> : List of extracted discrete curves in pixel coordinates 
  * - scale_factor : float                      : Scale factor from pixel to metric
  * <--------------- std::vector<DiscreteCurve> : List of resulting discrete curves. There are no guarantees of size,
                                                  number of curves or any correspondance with the original curves whatsoever */
std::vector<DiscreteCurve> filter_lines(std::vector<DiscreteCurve> const& branches, float scale_factor) {
	// Filter the extracted branches, and cut them where the angle gets too sharp, to avoid rough edges getting in the way of merging
	std::vector<DiscreteCurve> cut_lines;
	for (auto it = branches.begin(); it != branches.end(); it++) {
		// Too few points or too short (physically)
		if (it->size() <= config::lines::savgol_degree + 1 || it->length() <= config::lines::min_branch_length)
			continue;
		
		DiscreteCurve curve = *it;
		curve.resample(config::lines::branch_step);
		if (curve.size() <= config::lines::savgol_degree + 1)
			continue;
		
		std::vector<DiscreteCurve> cut_curve = cut_curve_angles(curve, config::lines::min_branch_length, config::lines::max_curvature_metric * scale_factor);
		cluster_lines.insert(cluster_lines.end(), cut_curve.begin(), cut_curve.end());
	}

	// Smooth the cuts
	std::vector<DiscreteCurve> cluster_lines;
	for (auto it = cut_lines.begin(); it != cut_lines.end(); it++) {
		DiscreteCurve curve = *it;
		if (curve.size() <= config::lines::savgol_degree + 1)
			continue;
		
		curve.savgol_filter(config::lines::initial_filter_window);
		cluster_lines.push_back(curve);
	}

	// Merge lines that are in the continuity of each other
	// This checks all the combinations of line and merge them when possible
	// This is a cache of combinations already checked, to avoid repeating work
	if (cluster_lines.size() >= 2) {
		std::set<std::pair<int, int>> combination_cache;
		std::set<std::pair<int, int>> new_cache;
		bool did_merge;
		do {
			did_merge = false;
			for (int index1 = 0; index1 < cluster_lines.size(); index1++) {
				int best_index = -1;
				MergeCandidate best_candidate;
				DiscreteCurve line1 = cluster_lines[index1];
				for (int index2 = index1 + 1; index2 < cluster_lines.size(); index2++) {
					if (combination_cache.find(std::pair<int, int>(index1, index2)) != combination_cache.end())
						continue;  // Aleady done, skip
					
					DiscreteCurve line2 = cluster_lines[index2];
					MergeCandidate candidate = check_mergeability(line1, line2);

					// The lines can be merged and they are the best candidates yet
					if (candidate.merge && (!best_candidate.merge || candidate.error * candidate.distance < best_candidate.error * best_candidate.distance)) {
						best_index = index2;
						best_candidate = candidate;
					}

					combination_cache.emplace(index1, index2);
				}

				
				if (!best_candidate.merge)
					continue;
				
				// There is a valid merge candidate : merge the curves
				// Remove the old lines and add the new, merged one
				DiscreteCurve line2 = cluster_lines[best_index];
				cluster_lines.erase(cluster_lines.begin() + best_index);
				cluster_lines.erase(cluster_lines.begin() + index1);

				// Join the curve with the right joint type
				DiscreteCurve merged_line;
				if (best_candidate.arc) merged_line = join_curves_arc(best_candidate, line1, line2, config::lines::branch_step);
				else                    merged_line = join_curves_line(best_candidate, line1, line2);
				cluster_lines.push_back(merged_line);

				// Remove the combinations that include the merged lines, to check them again
				// This should not be necessary, except for candidates that have been put away before and may be able to merge now
				// Also update the indices in the other combinations, as the merged lines have been removed
				new_cache.clear();
				for (auto it = combination_cache.begin(); it != combination_cache.end(); it++) {
					std::pair<int, int> combination = *it;
					if (combination.first == index1 || combination.second == index1 || combination.first == best_index || combination.second == best_index)
						continue;

					// Decrement the indices to account for the removal of lines at index `index1` and `best_index`
					// Decrement by 2 if it was after both lines, by 1 if it was after only one of them
					if (combination.first > index1 && combination.first > best_index)
						combination.first -= 2;
					else if (combination.first > index1 || combination.first > best_index)
						combination.first -= 1;

					if (combination.second > index1 && combination.second > best_index)
						combination.second -= 2;
					else if (combination.second > index1 || combination.second > best_index)
						combination.second -= 1;
					
					new_cache.insert(combination);
				}
				combination_cache = new_cache;

				// Break the current loop and start over with the new lines
				did_merge = true;
				break;
			}
		} while (did_merge);
	}

	// Smooth and filter the resulting lines, discard those that are too short
	for (auto it = cluster_lines.begin(); it != cluster_lines.end(); /* Done in the loop because of the erase */) {
		if (it->length() < config::lines::min_line_length) {
			it = cluster_lines.erase(it);
		} else {
			it->resample(config::lines::branch_step);
			it->savgol_filter(config::lines::smoothing_filter_window);
			it++;
		}
	}

	return cluster_lines;
}


// Precompute the gaussian filter kernel used in cut_curve_angles
constexpr std::array<float, config::lines::curvature_filter_size> _precompute_gaussian_kernel() {
	std::array<float, config::lines::curvature_filter_size> kernel = std::array<float, config::lines::curvature_filter_size>();
	float weight = 0;

	for (int i = 0; i < config::lines::curvature_filter_size; i++) {
		kernel[i] = std::exp(-sq(i - config::lines::curvature_filter_size / 2) / (2.0f * sq(config::lines::curvature_filter_deviation)));
		weight += kernel[i];
	}

	// Normalize to have the sum of the filter equal one
	for (int i = 0; i < config::lines::curvature_filter_size; i++) {
		kernel[i] /= weight;
	}

	return kernel;
}
const std::array<float, config::lines::curvature_filter_size> _cut_angles_gaussian_kernel = _precompute_gaussian_kernel();

/** Split a discrete curve where its curvature gets too high.
  * The curvature is calculated locally in rad/length unit, and smoothed using a gaussian filter
  * Here the measures are in "length units", this depends on the curve unit (pixels, meters, …)
  * - curve             : DiscreteCurve              : Discrete curve to split
  * - min_length        : float                      : Minimum length to keep a curve
  * - max_curvature     : float                      : Split the curve when its curvature gets over this threshold in rad/length unit
  * <-------------------- std::vector<DiscreteCurve> : List of resulting curve sections */
std::vector<DiscreteCurve> cut_curve_angles(DiscreteCurve const& curve, float min_length, float max_curvature) {
	// Compute the curvature at each point, that use the "second derivative" of the curve
	// Here the curvature in rad/unit is the relative angle between the incoming and outgoing vectors,
	// divided by the local mean length of a discrete curve segment
	arma::fvec curvature(curve.size() + config::lines::curvature_filter_size - 1);
	for (int i = 1; i < curve.size() - 1; i++) {
		arma::fvec prev_vector = curve.curve.col(i) - curve.curve.col(i - 1);
		arma::fvec next_vector = curve.curve.col(i + 1) - curve.curve.col(i);
		float local_curvature = 2 * vector_angle(prev_vector, next_vector) / (arma::norm(prev_vector) + arma::norm(next_vector));
		curvature(i + config::lines::curvature_filter_size / 2) = local_curvature;

		// Go to the ends of the curve
		if (i == 1)                curvature.head(config::lines::curvature_filter_size / 2 + 1).fill(local_curvature);
		if (i == curve.size() - 2) curvature.tail(config::lines::curvature_filter_size / 2 + 1).fill(local_curvature);
	}

	// Smooth the curvature data with a gaussian filter, that’s just a convolution
	curvature = arma::conv(curvature, arma::fvec(_cut_angles_gaussian_kernel.data(), config::lines::curvature_filter_size), "same");
	curvature = curvature.subvec(config::lines::curvature_filter_size / 2, curvature.n_elem - config::lines::curvature_filter_size / 2);

	/*arma::fvec gaussian_filter(_cut_angles_gaussian_kernel.data(), config::lines::curvature_filter_size);
	arma::fvec curvature(curve.size(), arma::fill::zeros);
	arma::fvec curvature_buffer(config::lines::curvature_filter_size);
	int curvature_index = 0;

	for (int i = 1; i < curve.size() - 1; i++) {
		arma::fvec prev_vector = curve.curve.col(i) - curve.curve.col(i - 1);
		arma::fvec next_vector = curve.curve.col(i + 1) - curve.curve.col(i);
		float local_curvature = 2 * vector_angle(prev_vector, next_vector) / (arma::norm(prev_vector) + arma::norm(next_vector));

		if (i == 1)
			curvature_buffer.fill(local_curvature);
		else
			curvature_buffer(curvature_index) = local_curvature;

		float smooth_curvature = 0.0f;
		for (int j = 0; j < config::lines::curvature_filter_size; j++)
			smooth_curvature += gaussian_filter(j) * curvature_buffer((curvature_index - (config::lines::curvature_filter_size - j - 1) + config::lines::curvature_filter_size) % config::lines::curvature_filter_size);

		curvature(i) = smooth_curvature;
		curvature_index = (curvature_index + config::lines::curvature_filter_size + 1) % config::lines::curvature_filter_size;
	}
	std::cout << curve.size() << " " << config::lines::curvature_filter_size << std::endl;
	curvature.t().print();*/


	// Now get the points that are on curvatures too sharp
	bool previous_in = false;
	int section_start = 0, section_end = 0;
	std::vector<DiscreteCurve> cut_curves;
	for (int i = 0; i < curve.size(); i++) {
		// Was ok, now bad -> end the section, split the curve
		if (curvature(i) > max_curvature && previous_in) {
			section_end = i;
			previous_in = false;
			DiscreteCurve split(curve.curve.cols(section_start, section_end - 1));
			if (split.size() > config::lines::savgol_degree + 1 && split.length() > min_length)
				cut_curves.push_back(split);
		}
		
		// Was bad, now ok -> start a section
		else if (curvature(i) <= max_curvature && !previous_in) {
			section_start = i;
			previous_in = true;
		}
	}

	// Tie up the loose ends : if a section started but didn’t end, it’s valid through to the end
	// If both `section_start` and `section_end` are still zero, no cuts took place and the whole curve is valid
	if (section_start > section_end) {
		DiscreteCurve split(curve.curve.cols(section_start, curve.size() - 1));
		if (split.size() > config::lines::savgol_degree + 1 && split.length() > min_length)
				cut_curves.push_back(split);
	}

	if (section_start == 0 && section_end == 0) {
		if (curve.size() > config::lines::savgol_degree + 1 && curve.length() > min_length)
			cut_curves.push_back(curve);
	}
	return cut_curves;
}

/** Check whether two curves can be joined, and how
  * First check whether a simple line joint can do the job, then with a circle arc
  * - line1 : DiscreteCurve  : First line to check
  * - line2 : DiscreteCurve  : Second line to check
  * <-------- MergeCandidate : Check result with all necessary information, check the structure definition */
MergeCandidate check_mergeability(DiscreteCurve const& line1, DiscreteCurve const& line2) {
	MergeCandidate candidate;

	// The lines could be in any direction
	// We need the points and vectors that directly face each other
	float extreme_distance_00 = arma::norm(line1.curve.col(0)                      - line2.curve.col(0));
	float extreme_distance_10 = arma::norm(line1.curve.col(line1.curve.n_cols - 1) - line2.curve.col(0));
	float extreme_distance_01 = arma::norm(line1.curve.col(0)                      - line2.curve.col(line2.curve.n_cols - 1));
	float extreme_distance_11 = arma::norm(line1.curve.col(line1.curve.n_cols - 1) - line2.curve.col(line2.curve.n_cols - 1));

	// To concatenate them at the end, the lines must be like 0 --line1--> -1 |----| 0 --line2--> -1
	// So the closest points must be line1[-1] and line2[0]
	// So flip 1 if the closest point is 0, flip 2 if the closest point is -1
	if (extreme_distance_00 <= extreme_distance_01 && extreme_distance_00 <= extreme_distance_10 && extreme_distance_00 <= extreme_distance_11) {
		candidate.distance = extreme_distance_00;
		candidate.flip1 = true;
		candidate.flip2 = false;
	} else if (extreme_distance_01 <= extreme_distance_00 && extreme_distance_01 <= extreme_distance_10 && extreme_distance_01 <= extreme_distance_11) {
		candidate.distance = extreme_distance_01;
		candidate.flip1 = true;
		candidate.flip2 = true;
	} else if (extreme_distance_10 <= extreme_distance_00 && extreme_distance_10 <= extreme_distance_01 && extreme_distance_10 <= extreme_distance_11) {
		candidate.distance = extreme_distance_10;
		candidate.flip1 = false;
		candidate.flip2 = false;
	} else {
		candidate.distance = extreme_distance_11;
		candidate.flip1 = false;
		candidate.flip2 = true;
	}

	// The closest points are too far away ⟶ ditch this combination
	if (candidate.distance > config::lines::merge_max_distance)
		return candidate;
	
	// Get the initial points and vectors, and the estimate index ranges depending on the flip status of both curves
	int start1, end1, start2, end2;
	arma::fvec point1, point2, vector1, vector2;

	if (candidate.flip1) {
		start1 = config::lines::estimate_start;
		end1 = config::lines::estimate_end;
		point1 = line1.curve.col(0);
		vector1 = point1 - line1.curve.col(1);
	} else {
		start1 = line1.size() - config::lines::estimate_end;
		end1 = line1.size() - config::lines::estimate_start;
		point1 = line1.curve.col(line1.size() - 1);
		vector1 = point1 - line1.curve.col(line1.size() - 2);
	}

	if (candidate.flip2) {
		start2 = line2.size() - config::lines::estimate_end;
		end2 = line2.size() - config::lines::estimate_start;
		point2 = line2.curve.col(line2.size() - 1);
		vector2 = -(point2 - line2.curve.col(line2.size() - 2));
	} else {
		start2 = config::lines::estimate_start;
		end2 = config::lines::estimate_end;
		point2 = line2.curve.col(0);
		vector2 = -(point2 - line2.curve.col(1));
	}

	// Get the vector from point1 to point2, and fail if the extreme vector of one of the curves differs too much from it
	arma::fvec joint_vector = point2 - point1;
	if (vector_angle(vector1, joint_vector) > config::lines::max_angle_diff || vector_angle(vector2, joint_vector) > config::lines::max_angle_diff)
		return candidate;
	
	// Clamp the estimate range to the amount of points
	if (start1 < 0) start1 = 0;
	if (start2 < 0) start2 = 0;
	if (end1 >= line1.size()) end1 = line1.size();
	if (end2 >= line2.size()) end2 = line2.size();

	// Try the line fitting
	fit_line(candidate, line1, line2, start1, end1, start2, end2);
	if (candidate.error < config::lines::max_rmse) {
		candidate.merge = true;
		candidate.arc = false;
	}

	// If it’s unsatisfactory, check with a circle arc
	fit_arc_kasa(candidate, line1, line2, start1, end1, start2, end2);
	if (candidate.error < config::lines::max_rmse) {
		candidate.merge = true;
		candidate.arc = true;
	}

	// Nope, nothing satisfactory, return with .merge = false
	return candidate;
}

/** Check mergeability by fitting a line for all points from line1[start1:end1] and line2[start2:end2]
  * - candidate    : MergeCandidate& : Merge check result to fill
  * - line1        : DiscreteCurve   : First curve to check
  * - line2        : DiscreteCurve   : Second curve to check
  * - start1, end1 : int             : Range of indices to use for the regression in line1 (start inclusive, end exclusive)
  * - start2, end2 : int             : Range of indices to use for the regression in line2 (start inclusive, end exclusive) */
void fit_line(MergeCandidate& candidate, DiscreteCurve const& line1, DiscreteCurve const& line2, int start1, int end1, int start2, int end2) {
	// As always, use PCA instead of the usual linear regression techniques
	// to get a linear regression in the general 2D plane
	arma::fvec eigenvalues;
	arma::fmat eigenvectors, points = arma::join_rows(line1.curve.cols(start1, end1 - 1), line2.curve.cols(start2, end2 - 1));
	if (!arma::eig_sym(eigenvalues, eigenvectors, arma::cov(points.t()))) {
		candidate.error = INFINITY;
		return;
	}

	// Then we can just transform the points in PCA space and compute the RMSE with the distances along the secondary component
	// To save us a bit of computation, we can just calculate the value along the secondary component
	arma::fvec ev2_values = (eigenvectors.row(1) * points).t();
	float ev2_mean = arma::mean(ev2_values);
	candidate.error = std::sqrt(arma::sum(arma::square(ev2_values - ev2_mean)) / points.n_cols);	
}

/** Fit a circular arc to all points in line1[start1:end1] and line2[start2:end2]
  * - candidate    : MergeCandidate& : Merge check result to fill
  * - line1        : DiscreteCurve   : First line to check
  * - line2        : DiscreteCurve   : Second line to check
  * - start1, end1 : int             : Index range to use for the regression on line1 (start inclusive, end exclusive)
  * - start2, end2 : int             : Index range to use for the regression on line2 (start inclusive, end exclusive) */
void fit_arc_kasa(MergeCandidate& candidate, DiscreteCurve const& line1, DiscreteCurve const& line2, int start1, int end1, int start2, int end2) {
	// Fit a circle arc with Kåsa’s method
	// We use this method because it is linear, and as such, much more efficient than the non-linear optimization methods
	// that require heavy iterative algorithms. Even though the results are not the same, it’s perfect for our use case

	// Compute the sums over all relevant points on both lines
	float sum_x = 0, sum_y = 0, sum_xx = 0, sum_yy = 0, sum_xxx = 0, sum_yyy = 0, sum_xy = 0, sum_xxy = 0, sum_xyy = 0;
	arma::fmat points = arma::join_rows(line1.curve.cols(start1, end1 - 1), line2.curve.cols(start2, end2 - 1));
	points.each_col([&](arma::fvec& point) {
		sum_x   += point(0);                   sum_y += point(1);
		sum_xx  += point(0)*point(0);          sum_xy += point(0)*point(1); sum_yy += point(1)*point(1);
		sum_xxx += point(0)*point(0)*point(0); sum_yyy += point(1)*point(1)*point(1);
		sum_xxy += point(0)*point(0)*point(1); sum_xyy += point(0)*point(1)*point(1);
	});

	// Intermediate results
	float alpha   = 2 * (sum_x * sum_x - points.n_cols * sum_xx);
	float beta    = 2 * (sum_x * sum_y - points.n_cols * sum_xy);
	float gamma   = 2 * (sum_y * sum_y - points.n_cols * sum_yy);
	float delta   = sum_xx*sum_x - points.n_cols*sum_xxx + sum_x*sum_yy - points.n_cols*sum_xyy;
	float epsilon = sum_xx*sum_y - points.n_cols*sum_yyy + sum_y*sum_yy - points.n_cols*sum_xxy;

	// Center of the circle
	candidate.center = {(delta*gamma - epsilon*beta) / (alpha*gamma - beta*beta),
	                    (alpha*epsilon - beta*delta) / (alpha*gamma - beta*beta)};
	
	// This method only gives the center of the circle
	// Infer the radius as the mean of distances to the center
	arma::frowvec distances = column_norm(points);
	candidate.radius = arma::mean(distances);

	// And finally, the RMS error with distance to center - radius as base error
	candidate.error = std::sqrt(arma::mean(arma::square(distances - candidate.radius)));
}


/** Join two curves with a line between their extreme points
  * - candidate : MergeCandidate : Merge check result on which to base the operation
  * - line1     : DiscreteCurve[M]   : First line to join
  * - line2     : DiscreteCurve[N]   : Second line to join
  * <------------ DiscreteCurve[M+N] : Resulting joined curve */
DiscreteCurve join_curves_line(MergeCandidate candidate, DiscreteCurve line1, DiscreteCurve line2) {
	// Just concatenate them, it creates a straight segment that will get resampled later
	DiscreteCurve joined(arma::join_rows(line1.curve, line2.curve));
	return joined;
}

/** Join two curves with a circle arc between the extreme points
  * - candidate   : MergeCandidate      : Merge check result with the arc parameters
  * - line1       : DiscreteCurve[M]    : First curve to join
  * - line2       : DiscreteCurve[N]    : Second curve to join
  * - branch_step : float               : Base distance between curve points for the interpolation
  * <-------------- float[2, N + x + M] : Joined and interpolated curve */
DiscreteCurve join_curves_arc(MergeCandidate candidate, DiscreteCurve line1, DiscreteCurve line2, float branch_step) {
	arma::fvec last_point1, last_point2;
	if (candidate.flip1) last_point1 = line1.curve.col(0);
	else                 last_point1 = line1.curve.col(line1.size() - 1);
	if (candidate.flip2) last_point2 = line2.curve.col(line2.size() - 1);
	else                 last_point2 = line2.curve.col(0);

	// Compute the vectors from the center of the circle to those extreme points
	// The interpolation arc is over the relative angle between those two vectors
	arma::fvec radial_vector1 = last_point1 - candidate.center;
	arma::fvec radial_vector2 = last_point2 - candidate.center;
	float joint_angle = vector_angle(radial_vector1, radial_vector2);

	// branch_step / radius is the angle step around that circle in radians
	// N angle steps -> N - 1 intermediate points
	// The joint points are just the center vectors rotated by the angle step, and with their lengths interpolated
	float angle_step = branch_step / candidate.radius;
	int num_points = int(joint_angle / angle_step - 1);
	float radial_vector1_norm = arma::norm(radial_vector1);
	float radial_vector2_norm = arma::norm(radial_vector2);
	
	arma::fmat joint(2, num_points);
	for (int i = 0; i < num_points; i++) {
		float rotation_angle = angle_step * (i + 1);
		float length_factor = (rotation_angle*radial_vector2_norm + (joint_angle - rotation_angle)*radial_vector1_norm) / (joint_angle * radial_vector1_norm);
		arma::fmat rotation_matrix = {{std::cos(rotation_angle), -std::sin(rotation_angle)},
		                              {std::sin(rotation_angle),  std::cos(rotation_angle)}};
		joint.col(i) = candidate.center + length_factor * rotation_matrix * radial_vector1;
	}

	DiscreteCurve joined(arma::join_rows(line1.curve, joint, line2.curve));
	return joined;
}
