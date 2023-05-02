#ifndef _TRAJECTORY_FISH2BIRD_H
#define _TRAJECTORY_FISH2BIRD_H

#include <tuple>
#include <armadillo>
#include <opencv2/opencv.hpp>


namespace fish2bird {
	/** Directly calculate the 3D coordinates in the target frame of the given points in input image pixel coordinates
	  * - image_points     : arma::fmat[2, N] : Pixel coordinates in the image (vectors as columns)
	  * - camera_to_image  : arma::fmat[3, 3] : Projection matrix (K) to pixel coordinates
	  * - camera_to_target : arma::fmat[4, 4] : Transform matrix from the camera frame to the target frame where all points’ Z coordinate is null
	  * - xi               : float            : Mei’s model center shifting parameter (ξ in his paper)
	  * <------------------- arma::fmat[3, N] : 3D coordinates of the points in the target frame. All Z coordinates are 0. */
	arma::fmat image_to_target(arma::fmat const& image_points, arma::fmat const& camera_to_image, arma::fmat const& camera_to_target, float xi);


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
	std::tuple<arma::fmat, float> target_to_output(arma::fmat const& target_points, float x_min, float x_max, float y_min, float y_max, int output_height, bool flip_x=false, bool flip_y=false);


	/** Directly create a bird-eye view from an image
	  * - image            : cv::Mat[CV_8U, y, x] : Original image
	  * - camera_to_image  : arma::fmat[3, 3]     : Projection matrix (K) to pixel coordinates
	  * - camera_to_target : arma::fmat[4, 4]     : Transform matrix from the camera frame to the target frame where all points’ Z coordinate is null
	  * - xi               : float                : Mei’s model center shifting parameter (ξ in his paper)
	  * - x_min, x_max     : float                : Metric range to display on the image’s x axis (for instance, [-25, 25] makes an image that shows all points within 25 meters on each side of the origin)
	  * - y_min, y_max     : float                : Metric range to display on the image’s y axis (for instance, [0, 50] makes an image that shows all points from the origin to 50 meters forward)
	  * - output_height    : int                  : Height of the output image in pixels. The width is calculated accordingly to scale points right
	  * - interpolate=true : bool                 : If `true`, apply bilinear interpolation to bird-eye view pixels, otherwise nearest-neighbor
	  * - flip_x=false     : bool                 : If `true`, reverse the X axis
	  * - flip_y=false     : bool                 : If `true`, reverse the Y axis
	  * <------------------- cv::Mat[v, u]        : Resulting bird-eye view image
	  * <------------------- float                : Scale factor from pixels to metric (pixels × scale_factor = metric. If flip_x or flip_y are set, don’t forget to flip it back beforhand, scale_factor * (height-y) or scale_factor * (width-x) */
	std::tuple<cv::Mat, float> to_birdeye(cv::Mat const& image, arma::fmat const& camera_to_image, arma::fmat const& camera_to_target,
	                                      float xi, float x_min, float x_max, float y_min, float y_max, int output_height,
										  bool interpolate=true, bool flip_x=false, bool flip_y=false);


	/** Convert pixel points on the bird-eye view to the actual 3D point in the target frame, using the same parameters given to `to_birdeye`
	  * - points        : arma::fmat[2, N] : Pixel coordinates in the bird-eye view to convert
	  * - x_min, x_max  : float            : Metric range displayed on the bird-eye view’s x axis
	  * - y_min, y_max  : float            : Metric range displayed on the bird-eye view’s y axis
	  * - output_height : int              : Height of the bird-eye view in pixels
	  * - flip_x=false  : bool             : Set to the value given to `to_birdeye`
	  * - flip_y=false  : bool             : Set to the value given to `to_birdeye`
	  * <---------------- arma::fmat[d, N] : Corresponding points in the target frame, as 3D column vectors /
	                                         2D for the _2d versions, omitting the Z coordinate */
	arma::fmat birdeye_to_target(arma::fmat const& points, float x_min, float x_max, float y_min, float y_max, int output_height, bool flip_x=false, bool flip_y=false);
	arma::fmat birdeye_to_target_2d(arma::fmat const& points, float x_min, float x_max, float y_min, float y_max, int output_height, bool flip_x=false, bool flip_y=false);
	arma::fmat birdeye_to_target_2d(arma::fmat&& points, float x_min, float x_max, float y_min, float y_max, int output_height, bool flip_x=false, bool flip_y=false);


	/** Projects 3D points from the target frame to image pixel coordinates, like the camera would
	  * - target_points    : arma::fmat[3, N] : 3D points in the target frame
	  * - target_to_camera : arma::fmat[4, 4] : Transform matrix from the target frame to the camera frame
	  * - camera_to_image  : arma::fmat[3, 3] : Projection matrix (K) to pixel coordinates
	  * - xi               : float            : Mei’s model center shifting parameter (ξ in his paper)
	  * <------------------- arma::fmat[2, N] : Projected points as image pixel coordinates. Those coordinates may not be integers nor within the actual image */
	arma::fmat target_to_image(arma::fmat const& target_points, arma::fmat const& target_to_camera, arma::fmat const& camera_to_image, float xi);
}


#endif