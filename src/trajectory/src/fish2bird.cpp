#include <cmath>

#include "fish2bird.h"

#define sq(val) ((val)*(val))

// FIXME : Everything is done in terms of row operations on column-major matrices

namespace fish2bird {
	inline arma::fmat affmul_direct(arma::fmat const& transform, arma::fmat const& points) {
		return arma::affmul(transform, points).eval().head_rows(points.n_rows);
	}

	inline arma::fmat affmul_direct(arma::fmat const& transform, arma::fmat&& points) {
		points = arma::affmul(transform, points).eval().head_rows(points.n_rows);
		return points;
	}

	/** Directly calculate the 3D coordinates in the target frame of the given points in input image pixel coordinates
	  * - image_points     : arma::fmat[2, N] : Pixel coordinates in the image (vectors as columns)
	  * - camera_to_image  : arma::fmat[3, 3] : Projection matrix (K) to pixel coordinates
	  * - camera_to_target : arma::fmat[4, 4] : Transform matrix from the camera frame to the target frame where all points’ Z coordinate is null
	  * - xi               : float            : Mei’s model center shifting parameter (ξ in his paper)
	  * <------------------- arma::fmat[3, N] : 3D coordinates of the points in the target frame. All Z coordinates are 0. */
	arma::fmat image_to_target(arma::fmat const& image_points, arma::fmat const& camera_to_image, arma::fmat const& camera_to_target, float xi) {
		arma::fmat inverse_camera_matrix = arma::inv(camera_to_image);
		const float a = camera_to_target(2, 0), b = camera_to_target(2, 1), c = camera_to_target(2, 2), d = camera_to_target(2, 3);

		// The following computations are all in 3D, do them in-place to avoid unnecessary memory overhead
		arma::fmat temp_points(3, image_points.n_cols);

		// Convert image pixel coordinates to metric coordinates on the sensor plane (inverse_camera_matrix @ (x, y))
		temp_points.head_rows(2) = affmul_direct(inverse_camera_matrix, image_points);

		// Project onto the unit sphere, using the equations in Christopher Mei’s paper
		arma::frowvec sphere_factors = arma::sum(arma::square(temp_points.head_rows(2)), 0);
		sphere_factors = (xi + arma::sqrt(1 + (1 - sq(xi)) * sphere_factors)) / (sphere_factors + 1);
		temp_points.row(0) %= sphere_factors;
		temp_points.row(1) %= sphere_factors;
		temp_points.row(2) = sphere_factors - xi;

		// Basically, convert cartesian coordinates at the surface of the unit sphere to spheric coordinates
		// But as those are arccos and arctans and we only reuse the sin() and cos() of those values afterwards,
		// we can save some time and complexity by calculating those subresults directly
		// The temp_points layout is [cos_longitude, sin_longitude, sphere_z]
		arma::frowvec sphere_rnorms = 1 / arma::sqrt(arma::sum(arma::square(temp_points.head_rows(2))));
		temp_points.row(0) %= sphere_rnorms;
		temp_points.row(1) %= sphere_rnorms;
		arma::frowvec sin_colatitude = arma::sqrt(1 - arma::square(temp_points.row(2)));

		// Solve the projection equations to get the cartesian point in the camera frame using the spheric coordinates and the target plane
		arma::frowvec camera_radii = -d / (a * sin_colatitude % temp_points.row(0) + b * sin_colatitude % temp_points.row(1) + c * temp_points.row(2));
		temp_points.row(0) %= camera_radii % sin_colatitude;
		temp_points.row(1) %= camera_radii % sin_colatitude;
		temp_points.row(2) %= camera_radii;

		// Project to the target frame (matrix product camera_to_target × (camera_x, camera_y, camera_z))
		// The Z coordinates are set to 0 by definition
		temp_points = affmul_direct(camera_to_target, temp_points);
		temp_points.row(2) = 0;

		return temp_points;
	}


	/** Convert 3D points in the target space to bird-eye view pixel coordinates
	  * Make an orthogonal projection relative to the target Z axis (so parallel to the target plane)
	  * Warning : the resulting coordinates are not rounded to integers and may lie outside of the image
	  * - target_points : arma::fmat[3, N] : 3D coordinates of the points in the target frame
	  * - x_min, x_max  : float            : Metric range to display on the image’s x axis (for instance, [-25, 25] makes an image that shows all points within 25 meters on each side of the origin)
	  * - y_min, y_max  : float            : Metric range to display on the image’s y axis (for instance, [0, 50] makes an image that shows all points from the origin to 50 meters forward)
	  * - output_height : int              : Height of the output image in pixels. The width is calculated accordingly to scale points right
	  * - flip_x=false  : bool             : If `true`, reverse the X axis
	  * - flip_y=false  : bool             : If `true`, reverse the Y axis
	  * <---------------- arma::fmat[2, N] : Pixel coordinates of the given points for the output image described by the parameters
	  * <---------------- float            : Scale factor from pixels to metric (pixels × scale_factor = metric. If flip_x or flip_y are set, don’t forget to flip it back beforehand, scale_factor * (height-y) or scale_factor * (width-x) */
	std::tuple<arma::fmat, float> target_to_output(arma::fmat const& target_points, float x_min, float x_max, float y_min, float y_max, int output_height, bool flip_x, bool flip_y) {
		const float scale_factor = (y_max - y_min) / output_height;
		const int output_width = int((x_max - x_min) / scale_factor);

		// This is just scaling metric ⟶ pixel coordinates, and flipping if needed
		arma::fmat output_points(2, target_points.n_cols);
		output_points.row(0) = (target_points.row(0) - x_min) / scale_factor;
		output_points.row(1) = (target_points.row(1) - y_min) / scale_factor;
		
		if (flip_x)
			output_points.row(0) = output_width - output_points.row(0);
		if (flip_y)
			output_points.row(1) = output_height - output_points.row(1);
		
		return {output_points, scale_factor};
	}


	/** Directly create a bird-eye view from an image
	  * - image            : cv::Mat[CV_8U, y, x] : Original image
	  * - camera_to_image  : arma::fmat[3, 3]     : Projection matrix (K) to pixel coordinates
	  * - target_to_camera : arma::fmat[4, 4]     : Transform matrix from the target frame where all points’ Z coordinate is null to the camera frame
	  * - xi               : float                : Mei’s model center shifting parameter (ξ in his paper)
	  * - x_min, x_max     : float                : Metric range to display on the image’s x axis (for instance, [-25, 25] makes an image that shows all points within 25 meters on each side of the origin)
	  * - y_min, y_max     : float                : Metric range to display on the image’s y axis (for instance, [0, 50] makes an image that shows all points from the origin to 50 meters forward)
	  * - output_height    : int                  : Height of the output image in pixels. The width is calculated accordingly to scale points right
	  * - interpolate=true : bool                 : If `true`, apply bilinear interpolation to bird-eye view pixels, otherwise nearest-neighbor
	  * - flip_x=false     : bool                 : If `true`, reverse the X axis
	  * - flip_y=false     : bool                 : If `true`, reverse the Y axis
	  * <------------------- cv::Mat[v, u]        : Resulting bird-eye view image
	  * <------------------- float                : Scale factor from pixels to metric (pixels × scale_factor = metric. If flip_x or flip_y are set, don’t forget to flip it back beforhand, scale_factor * (height-y) or scale_factor * (width-x) */
	std::tuple<cv::Mat, float> to_birdeye(cv::Mat const& image, arma::fmat const& camera_to_image, arma::fmat const& target_to_camera,
	                                      float xi, float x_min, float x_max, float y_min, float y_max, int output_height,
										  bool interpolate, bool flip_x, bool flip_y) {
		const float scale_factor = (y_max - y_min) / output_height;
		const int output_width = int((x_max - x_min) / scale_factor);

		cv::Mat birdeye = cv::Mat::zeros(output_height, output_width, CV_8U);

		// Here the problem is kinda reversed : we take each point of the final bird-eye view,
		// and see which point of the original image it projects to
		// Then that bird-eye view pixel is assigned the value at that point of the original image, with bilinear interpolation if needed
		// Here everything is vectorized, that takes more memory but it’s presumably more efficient
		// Also, it’s all done mostly in a single buffer to avoid unnecessary memory overhead
		arma::umat output_points(2, output_width * output_height);
		arma::fmat temp_points(3, output_width * output_height);
		
		// Initialize with the output pixel coordinates
		for (int y = 0; y < output_height; y++) {
			for (int x = 0; x < output_width; x++) {
				output_points(0, y*output_width + x) = x;
				output_points(1, y*output_width + x) = y;
			}
		}

		// Get the metric points on the target plane corresponding to those pixel coordinates
		// temp_points is [x, y, z=0]
		if (flip_x) temp_points.row(0) = arma::conv_to<arma::frowvec>::from(output_width - output_points.row(0)) * scale_factor + x_min;
		else        temp_points.row(0) = arma::conv_to<arma::frowvec>::from(output_points.row(0)) * scale_factor + x_min;
		if (flip_y) temp_points.row(1) = arma::conv_to<arma::frowvec>::from(output_height - output_points.row(1)) * scale_factor + y_min;
		else        temp_points.row(1) = arma::conv_to<arma::frowvec>::from(output_points.row(1)) * scale_factor + y_min;
		temp_points.row(2).fill(0);

		// Transformation to camera frame, target_to_camera × camera_points
		// temp_points is [x, y, z] in the camera frame
		temp_points = affmul_direct(target_to_camera, temp_points);

		// Discard points with Zc < 0, since they are behind the camera thus can’t be on the image
		arma::uvec back_filter = arma::find(temp_points.row(2) < 0);
		temp_points.shed_cols(back_filter);
		output_points.shed_cols(back_filter);

		// Project on the sensor plane
		// temp_points is [sensor_x, sensor_y, projection_norm]
		temp_points.row(2) = 1 / (temp_points.row(2) + xi * arma::sqrt(arma::sum(arma::square(temp_points), 0)));
		temp_points.row(0) %= temp_points.row(2);
		temp_points.row(1) %= temp_points.row(2);

		// Convert to pixel coordinates (matrix product, camera_to_image × (sensor_x, sensor_y))
		temp_points.head_rows(2) = affmul_direct(camera_to_image, temp_points.head_rows(2));

		// Discard the points that are outside of the original image
		arma::uvec range_filter = arma::find(0 <= temp_points.row(0) && temp_points.row(0) < image.cols && 0 <= temp_points.row(1) && temp_points.row(1) < image.rows);
		for (int i = 0; i < range_filter.n_elem; i++) {
			int index = range_filter(i);
			int output_x = output_points(0, index), output_y = output_points(1, index);
			float image_x = temp_points(0, index), image_y = temp_points(1, index);

			// Without interpolation, if disabled or on the edges of the image
			if (!interpolate || std::ceil(image_x) >= image.cols || std::ceil(image_y) >= image.rows || std::floor(image_x) == image_x) {
				birdeye.at<uint8_t>(output_y, output_x) = image.at<uint8_t>(int(std::floor(image_y)), int(std::floor(image_x)));
			}

			// With bilinear interpolation
			else {
				float x_factor = std::ceil(image_x) - image_x;
				float y_factor = std::ceil(image_y) - image_y;
				birdeye.at<uint8_t>(output_y, output_x) = 
					uint8_t(     x_factor  * (     y_factor  * image.at<uint8_t>(int(floor(image_y)), int(floor(image_x))) +
					                          (1 - y_factor) * image.at<uint8_t>(int( ceil(image_y)), int(floor(image_x)))) +
					        (1 - x_factor) * (     y_factor  * image.at<uint8_t>(int(floor(image_y)), int( ceil(image_x))) +
					                          (1 - y_factor) * image.at<uint8_t>(int( ceil(image_y)), int( ceil(image_x)))));
			}
		}

		return {birdeye, scale_factor};
	}


	/** Convert pixel points on the bird-eye view to the actual 3D point in the target frame, using the same parameters given to `to_birdeye`
	  * - points        : arma::fmat[2, N] : Pixel coordinates in the bird-eye view to convert
	  * - x_min, x_max  : float            : Metric range displayed on the bird-eye view’s x axis
	  * - y_min, y_max  : float            : Metric range displayed on the bird-eye view’s y axis
	  * - output_height : int              : Height of the bird-eye view in pixels
	  * - flip_x=false  : bool             : Set to the value given to `to_birdeye`
	  * - flip_y=false  : bool             : Set to the value given to `to_birdeye`
	  * <---------------- arma::fmat[d, N] : Corresponding points in the target frame, as 3D column vectors /
	                                         2D for the _2d versions */
	arma::fmat birdeye_to_target(arma::fmat const& points, float x_min, float x_max, float y_min, float y_max, int output_height, bool flip_x, bool flip_y) {
		const float scale_factor = (y_max - y_min) / output_height;
		const int output_width = int((x_max - x_min) / scale_factor);

		// This just involves scaling and flipping when necessary
		arma::fmat target_points(3, points.n_cols);
		
		if (flip_x) target_points.row(0) = (output_width - points.row(0)) * scale_factor + x_min;
		else        target_points.row(0) = points.row(0) * scale_factor + x_min;
		if (flip_y) target_points.row(1) = (output_height - points.row(1)) * scale_factor + y_min;
		else        target_points.row(1) = points.row(1) * scale_factor + y_min;
		target_points.row(2) = 0;
		
		return target_points;
	}

	/** 2D version */
	arma::fmat birdeye_to_target_2d(arma::fmat const& points, float x_min, float x_max, float y_min, float y_max, int output_height, bool flip_x, bool flip_y) {
		const float scale_factor = (y_max - y_min) / output_height;
		const int output_width = int((x_max - x_min) / scale_factor);

		// This just involves scaling and flipping when necessary
		arma::fmat target_points(2, points.n_cols);
		
		if (flip_x) target_points.row(0) = (output_width - points.row(0)) * scale_factor + x_min;
		else        target_points.row(0) = points.row(0) * scale_factor + x_min;
		if (flip_y) target_points.row(1) = (output_height - points.row(1)) * scale_factor + y_min;
		else        target_points.row(1) = points.row(1) * scale_factor + y_min;
		
		return target_points;
	}

	/** 2D version with move semantic, do it in place and return the same object*/
	arma::fmat birdeye_to_target_2d(arma::fmat&& points, float x_min, float x_max, float y_min, float y_max, int output_height, bool flip_x, bool flip_y) {
		const float scale_factor = (y_max - y_min) / output_height;
		const int output_width = int((x_max - x_min) / scale_factor);

		// This just involves scaling and flipping when necessary		
		if (flip_x) points.row(0) = (output_width - points.row(0)) * scale_factor + x_min;
		else        points.row(0) = points.row(0) * scale_factor + x_min;
		if (flip_y) points.row(1) = (output_height - points.row(1)) * scale_factor + y_min;
		else        points.row(1) = points.row(1) * scale_factor + y_min;
		
		return points;
	}


	/** Projects 3D points from the target frame to image pixel coordinates, like the camera would
	  * - target_points    : arma::fmat[3, N] : 3D points in the target frame
	  * - target_to_camera : arma::fmat[4, 4] : Transform matrix from the target frame to the camera frame
	  * - camera_to_image  : arma::fmat[3, 3] : Projection matrix (K) to pixel coordinates
	  * - xi               : float            : Mei’s model center shifting parameter (ξ in his paper)
	  * <------------------- arma::fmat[2, N] : Projected points as image pixel coordinates. Those coordinates may not be integers nor within the actual image */
	arma::fmat target_to_image(arma::fmat const& target_points, arma::fmat const& target_to_camera, arma::fmat const& camera_to_image, float xi) {
		// All this is mostly done in a single buffer to avoid unnecessary memory overhead
		// The current temp_points configuration is [x, y, z] in the camera frame 
		arma::fmat temp_points = affmul_direct(target_to_camera, target_points);
		arma::frowvec camera_rnorms = 1 / arma::sqrt(arma::sum(arma::square(temp_points), 0));

		// Project on the unit sphere
		// temp_points is [sphere_x, sphere_y, sphere_biased_z]
		temp_points.row(0) %= camera_rnorms;
		temp_points.row(1) %= camera_rnorms;
		temp_points.row(2) = 1 / (temp_points.row(2) % camera_rnorms + xi);

		// Project to the sensor plane
		// temp_points is [sensor_x, sensor_y]
		temp_points.row(0) %= temp_points.row(2);
		temp_points.row(1) %= temp_points.row(2);
		temp_points = temp_points.head_rows(2);

		// Convert to pixel coordinates
		return affmul_direct(camera_to_image, temp_points);
	}
}