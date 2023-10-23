#include <mlpack/methods/dbscan.hpp>

#include "trajectory/Marking.h"


/** Extract road markings when possible. Branches used by those markings are removed from the `branches` vector
  * - branches        : std::vector<DiscreteCurve>& : List of discrete curves detected in the image
  * - scale_factor    : float                       : Scale factor from pixel to metric in the bird-eye view
  * <------------------ std::vector<Marking>        : Detected markings */
std::vector<Marking> detect_markings(std::vector<DiscreteCurve>& branches, float scale_factor) {
	std::vector<Marking> markings;
	if (branches.size() == 0)
		return markings;
	
	detect_crosswalks(markings, branches, scale_factor);
	return markings;
}


void detect_crosswalks(std::vector<Marking>& markings, std::vector<DiscreteCurve>& branches, float scale_factor) {
	std::vector<int> band_indices;
	std::vector<arma::fvec> crosswalk_vectors;
	std::vector<arma::fmat> band_rectangles;
	for (int i = 0; i < branches.size(); i++) {
		DiscreteCurve branch = branches[i];
		branch.resample(5);
		if (branch.size() < 3)
			continue;
		branch.curve = birdeye_to_target_config(branch.curve);
		
		// Perform the eigendecomposition
		// The eigenvalues are in ascending order and the associated eigenvectors are column vectors
		arma::fvec eigenvalues;
		arma::fmat eigenvectors;
		if (!arma::eig_sym(eigenvalues, eigenvectors, arma::cov(branch.curve.t())))
			continue;
				
		// The covariance is singular ⟶ all points are on a perfect line
		// That’s cool but it makes quite a lousy rectangle
		if (arma::any(eigenvalues == 0))
			continue;
		
		// Take the angle of the principal component = the highest eigenvalue = the last here
		// Note that in all that follows, ev1 is the principal component = row(1), ev2 is the smaller one = row(0)
		float main_angle = vector_angle(eigenvectors.col(1));

		// Transform all points to PCA space
		arma::fmat pca_points = eigenvectors.t() * branch.curve;
		arma::fvec ev_min = arma::min(pca_points, 1);
		arma::fvec ev_max = arma::max(pca_points, 1);

		// Null width or height : nothing interesting here
		if (arma::any(ev_min == ev_max))
			continue;

		// Divide the PCA space into the regions considered for each edge of the rectangle
		float edge_interval = (ev_max(0) - ev_min(0)) * config::markings::crosswalk::ev2_edge_part;
		float max_ev2_bottom = ev_min(0) + edge_interval;
		float min_ev2_top    = ev_max(0) - edge_interval;
		float max_ev1_left   = ev_min(1) + edge_interval;
		float max_ev1_right  = ev_max(1) - edge_interval;

		// We are in PCA space, so this is easy : the rectangle is perfectly horizontal
		// Take the median of each edge’s region to get the ev1 and ev2 coordinates of the edges of the rectangle
		float ev2_bottom = arma::median(arma::frowvec(pca_points.row(0))(arma::find(pca_points.row(0) <= max_ev2_bottom)));
		float ev2_top    = arma::median(arma::frowvec(pca_points.row(0))(arma::find(pca_points.row(0) >= min_ev2_top   )));
		float ev1_left   = arma::median(arma::frowvec(pca_points.row(1))(arma::find(pca_points.row(1) <= max_ev1_left  )));
		float ev1_right  = arma::median(arma::frowvec(pca_points.row(1))(arma::find(pca_points.row(1) >= max_ev1_right )));

		float rect_width = ev1_right - ev1_left;
		float rect_height = ev2_top - ev2_bottom;

		// Now compute the sum of squared distances of the fitted rectangle to the actual curve, still in PCA space
		// The distance of a curve point to the rectangle is its distance to the closest edge of the rectangle
		// And as those edges are just vertical and horizontal lines in PCA space, all this is easy to compute
		arma::fvec errors = arma::min(arma::min(arma::square(pca_points.row(0) - ev2_bottom),
		                                        arma::square(pca_points.row(0) - ev2_top)),
		                              arma::min(arma::square(pca_points.row(1) - ev1_left),
		                                        arma::square(pca_points.row(1) - ev1_right))).t();
		
		// Compute the root mean squared error, normalized by the rectangle area, otherwise it’s not fair to larger rectangles
		float error_rmse = std::sqrt(arma::sum(errors) / (/*errors.n_elem * */(rect_width * rect_height)));

		// Only keep relatively good rectangles that are at least a certain size
		if (error_rmse < config::markings::crosswalk::max_rmse &&
					config::environment::crosswalk_width * (1 - config::markings::size_tolerance) < rect_height &&
					config::environment::crosswalk_width * (1 + config::markings::size_tolerance) > rect_height) {
			// Compute the corners in PCA space, as the rectangle is orthogonal to the base it’s way easier there
			arma::fmat pca_corners = {{ev2_top , ev2_top  , ev2_bottom, ev2_bottom},
			                          {ev1_left, ev1_right, ev1_right , ev1_left  }};

			// Back to normal space
			arma::fmat rect_corners = eigenvectors * pca_corners;
			arma::fvec centroid = arma::mean(rect_corners, 1);

			// Now, to detect crosswalks and dotted lines, we project the center of the rectangle onto its eigenvectors
			// The direct projection is along the principal component, as crosswalk rectangles have their largest dimension parallel,
			// their centers project approximately at the same place on their principal component
			float direct_projection = arma::dot(eigenvectors.col(1), centroid) / arma::dot(eigenvectors.col(1), eigenvectors.col(1));
			arma::fvec band_data = {rect_width / (2 * config::birdeye::x_range),                                                    // Height, normalized by the region
			                        (rect_height - config::environment::crosswalk_width) / config::environment::crosswalk_width,    // Width error relative to a standard crosswalk band width
									main_angle / arma::fdatum::pi,                                                                  // Main angle, normalized by π
									direct_projection / (config::birdeye::y_range - config::birdeye::roi_y)};                       // Projection on the principal component, normalized by the region
			crosswalk_vectors.emplace_back(std::move(band_data));
			band_rectangles.emplace_back(std::move(rect_corners));
			band_indices.push_back(i);
		}
	}

	// Not enough bands to be an actual crosswalk
	if (crosswalk_vectors.size() < config::markings::crosswalk::min_bands)
		return;

	// Now try to cluster the bands we found into crosswalks
	// Crosswalks are rows of rectangles, with approximately the same shape and angle
	// So cluster the bands with a DBSCAN and see what comes out
	arma::dmat crosswalk_data(4, crosswalk_vectors.size());
	for (int i = 0; i < crosswalk_vectors.size(); i++)
		crosswalk_data.col(i) = arma::conv_to<arma::dvec>::from(crosswalk_vectors[i]);

	arma::Row<size_t> assignments;
	mlpack::DBSCAN dbscan(config::markings::crosswalk::dbscan_epsilon, config::markings::crosswalk::min_bands);
	dbscan.Cluster<arma::dmat>(crosswalk_data, assignments);
	arma::Row<size_t> available_clusters = arma::unique(assignments);
	available_clusters.for_each([&](size_t const& cluster) {
		if (cluster == SIZE_MAX)
			return;
		
		arma::uvec cluster_indices = arma::find(assignments == cluster);
		int num_bands = cluster_indices.n_elem;

		// FIXME : For the moment, confidence just depends on the amount of bands detected
		Marking marking(Marking::Type::Crosswalk, std::min(1.0f, (float)cluster_indices.n_elem / 6), 2, 4, num_bands);
		for (int i = 0; i < cluster_indices.n_elem; i++) {
			int band_index = cluster_indices(i);
			int global_index = band_indices[band_index];
			marking.data.slice(i) = band_rectangles[band_index];
			branches.erase(branches.begin() + global_index);
		}

		markings.push_back(marking);
	});
}