#include "fish2bird.h"
#include "trajectory/TrajectoryExtractorNode.h"


//                         ╔══════════════════╗                          #
// ════════════════════════╣ IMAGE PROCESSING ╠═════════════════════════ #
//                         ╚══════════════════╝                          #

/** Preprocess the image retrieved from the camera
  * - image            : cv::Mat[CV_8UC3, y, x] : RGB image received from the camera
  * - target_to_camera : arma::fmat[4, 4]       : 3D homogeneous transform matrix from the target (road) frame to the camera frame
  * <------------------- cv::Mat[CV_8U, v, u]   : Full grayscale bird-eye view (mostly for visualisation)
  * <------------------- ndarray[CV_8U, v, u]   : Fully preprocessed bird-eye view (binarized, edge-detected)
  * <------------------- float                  : Scale factor, multiply by this to convert lengths from pixel to metric in the target frame */
std::tuple<cv::Mat, cv::Mat, float> TrajectoryExtractorNode::preprocess_image(cv::Mat& image, arma::fmat const& target_to_camera) {
	// Convert to grayscale
	cv::Mat gray_image(image.rows, image.cols, CV_8U);
	cv::cvtColor(image, gray_image, cv::COLOR_RGB2GRAY);
	
	// First a gaussian blur is applied to reduce noise (in-place)
	// cv::GaussianBlur(gray_image, gray_image, cv::Size(7, 7), 1.5);

	// Project to bird-eye view
	auto [birdeye, scale_factor] = fish2bird::to_birdeye(gray_image, m_camera_to_image, target_to_camera, m_distortion_xi,
	                                                    -config::birdeye::x_range, config::birdeye::x_range, config::birdeye::roi_y, config::birdeye::y_range,
											            config::birdeye::birdeye_size, true, false, true);

	// Gaussian adaptive thresholding to reduce the influence of lighting changes
	cv::Mat be_binary(birdeye.rows, birdeye.cols, CV_8U);
	cv::adaptiveThreshold(birdeye, be_binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, config::preprocess::threshold_window, config::preprocess::threshold_bias);
	// cv::imwrite("threshold.png", be_binary);

	// The adaptive binarization makes the borders white, mask them out
	cv::Mat mask(birdeye.rows, birdeye.cols, CV_8U);
	cv::Mat mask_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(config::preprocess::threshold_window / 2 + 2, config::preprocess::threshold_window / 2 + 2));
	cv::threshold(birdeye, mask, 0, 255, cv::THRESH_BINARY);
	cv::erode(mask, mask, mask_kernel);
	cv::bitwise_and(be_binary, mask, be_binary);

	// Binarize the image relative to an ambiance map based upon a median filter
	// cv::Mat ambiance(birdeye.rows, birdeye.cols, CV_8U);
	// cv::medianBlur(birdeye, ambiance, config::preprocess::ambiance_filter_size);
	// ambiance *= config::preprocess::ambiance_bias;

	// cv::Mat be_binary(birdeye.rows, birdeye.cols, CV_8U);
	// cv::compare(birdeye, ambiance, be_binary, cv::CMP_GT);

	// Apply an opening operation to eliminate a few artifacts and better separate blurry markings
	cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(config::preprocess::open_kernel_size, config::preprocess::open_kernel_size));
	cv::morphologyEx(be_binary, be_binary, cv::MORPH_OPEN, open_kernel);

	// Edge detection to get the 1-pixel wide continuous curves required by the following operations
	cv::Canny(be_binary, be_binary, 50, 100);

	return {birdeye, be_binary, scale_factor};
}
