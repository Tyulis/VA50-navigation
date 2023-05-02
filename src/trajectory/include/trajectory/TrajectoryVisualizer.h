#ifndef _TRAJECTORY_TRAJECTORYVISUALIZER_H
#define _TRAJECTORY_TRAJECTORYVISUALIZER_H

#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

#include "trajectory/Marking.h"
#include "trajectory/DiscreteCurve.h"


// Those are the matplotlib default colors
constexpr int default_color_scheme_size = 10;
constexpr uint8_t default_color_scheme[default_color_scheme_size][3] = {
	{0x1F, 0x77, 0xB4}, {0xFF, 0x7F, 0x0E},
	{0x2C, 0xA0, 0x2C}, {0xD6, 0x27, 0x28},
	{0x94, 0x67, 0xBD}, {0x8C, 0x56, 0x4B},
	{0xE3, 0x77, 0xC2}, {0x7F, 0x7F, 0x7F},
	{0xBC, 0xBD, 0x22}, {0x17, 0xBE, 0xCF},
};

inline cv::Scalar color_scheme_value(const uint8_t color_scheme[][3], int color_scheme_size, int index) {
	return cv::Scalar(color_scheme[index % color_scheme_size][0], color_scheme[index % color_scheme_size][1], color_scheme[index % color_scheme_size][2]);
}

// With just the index, give the default color scheme
inline cv::Scalar color_scheme_value(int index) {
	return cv::Scalar(default_color_scheme[index % default_color_scheme_size][0], default_color_scheme[index % default_color_scheme_size][1], default_color_scheme[index % default_color_scheme_size][2]);
}


class TrajectoryVisualizer {
	public:
		TrajectoryVisualizer() = default;

		void update_background(cv::Mat const& be_binary);
		void update_line_detection(std::vector<DiscreteCurve> const& lines, int left_line_index, int right_line_index, std::vector<Marking> const& markings, float scale_factor);
		void update_trajectory_construction(cv::Mat const& viz);
	
	private:
		void update();

		cv::Mat m_line_viz;
		cv::Mat m_trajectory_viz;
		cv::Mat m_full_viz;
};

#endif