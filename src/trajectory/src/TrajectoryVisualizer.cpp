#include <algorithm>

#include "trajectory/Utility.h"
#include "trajectory/TrajectoryVisualizer.h"



/** Set the background image for the next line detection update
  * - be_binary        : cv::Mat[y, x, CV_8U]       : Preprocessed camera image (binary edge-detected bird-eye view) */
void TrajectoryVisualizer::update_background(cv::Mat const& be_binary) {
	if (m_line_viz.empty())
		m_line_viz = cv::Mat(be_binary.rows, be_binary.cols, CV_8UC3);
	cv::cvtColor(be_binary, m_line_viz, cv::COLOR_GRAY2RGB);
}

/** Generate and update the left visualization from the preprocessed image and the detected lines and markings
  * - lines            : std::vector<DiscreteCurve> : Detected discrete curves in the image
  * - left_line_index  : int                        : Index of the left lane marking in the `lines` list, or -1
  * - right_line_index : int                        : Index of the right lane marking in the `lines` list, or -1
  * - markings         : std::vector<Marking>       : List of detected road markings. Currently only supports MarkingType::Crosswalk */
void TrajectoryVisualizer::update_line_detection(std::vector<DiscreteCurve> const& lines, int left_line_index, int right_line_index, std::vector<Marking> const& markings, float scale_factor) {
	if (lines.size() > 0) {
		// Show the detected curves
		int max_points = std::max_element(lines.begin(), lines.end(), [](DiscreteCurve const& lhs, DiscreteCurve const& rhs) {
			return lhs.size() < rhs.size();
		})->size();

		for (int i = 0; i < lines.size(); i++) {
			std::vector<cv::Point> linepoints = arma::conv_to<std::vector<cv::Point>>::from(target_to_birdeye_config(lines[i].curve));
			
			if (left_line_index >= 0 && i == left_line_index)
				cv::polylines(m_line_viz, linepoints, false, cv::Scalar(255, 0, 0), 4);
			else if (right_line_index >= 0 && i == right_line_index)
				cv::polylines(m_line_viz, linepoints, false, cv::Scalar(0, 100, 255), 4);
			else
				cv::polylines(m_line_viz, linepoints, false, cv::Scalar(0, 200, 0), 2);
		}

		// Draw the crosswalks
		for (int i = 0; i < markings.size(); i++) {
			if (markings[i].type == Marking::Type::Crosswalk) {
				cv::Scalar color = color_scheme_value(i);
				std::vector<std::vector<cv::Point>> band_points;

				// Each slice is one rectangular band of the crosswalk
				markings[i].data.each_slice([&](arma::fmat const& band) {
					arma::imat band_be = target_to_birdeye_config(band);					
					band_points.emplace_back(arma::conv_to<std::vector<cv::Point>>::from(band_be));
				});
				cv::fillPoly(m_line_viz, band_points, color);
			}
		}
	}

	cv::cvtColor(m_line_viz, m_line_viz, cv::COLOR_RGB2BGR);
	update();
}

/** Update the right visualization with the given HSV image */
void TrajectoryVisualizer::update_trajectory_construction(cv::Mat const& viz) {
	if (m_trajectory_viz.empty())
		m_trajectory_viz = cv::Mat(viz.rows, viz.cols, CV_8UC3);
	cv::cvtColor(viz, m_trajectory_viz, cv::COLOR_HSV2BGR);
	update();
}

/** Update the visualization window */
void TrajectoryVisualizer::update() {
	if (m_line_viz.empty() || m_trajectory_viz.empty())
		return;
	
	// Merge both images
	if (m_full_viz.empty())
		m_full_viz = cv::Mat(std::max(m_line_viz.rows, m_trajectory_viz.rows), m_line_viz.cols + m_trajectory_viz.cols, CV_8UC3);
	
	m_line_viz.copyTo(m_full_viz(cv::Rect(0, 0, m_line_viz.cols, m_line_viz.rows)));
	m_trajectory_viz.copyTo(m_full_viz(cv::Rect(m_line_viz.cols, 0, m_trajectory_viz.cols, m_trajectory_viz.rows)));

	cv::imshow("viz", m_full_viz);
	cv::waitKey(1);
}