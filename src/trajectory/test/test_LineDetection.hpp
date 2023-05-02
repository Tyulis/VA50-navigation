#include <iostream>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "trajectory/LineDetection.h"

#include "testref_resample.h"
#include "testref_savgol.h"
#include "testref_branches.h"
#include "testref_length.h"
#include "testref_cut_angles.h"
#include "testref_filter_lines.h"

TEST(TestLineDetection, extract_branches) {
	cv::Mat image = cv::imread("../../testref-be-binary.png", cv::IMREAD_GRAYSCALE);
	std::vector<arma::fmat> ref_branches = get_testref_branches();

	std::vector<DiscreteCurve> branches = extract_branches(image);

	EXPECT_EQ(ref_branches.size(), branches.size());

	for (int i = 0; i < branches.size(); i++) {
		if (i >= ref_branches.size())
			continue;
		
		bool equal = arma::approx_equal(branches[i].curve, ref_branches[i], "absdiff", 0.001);
		if (!equal) {
			std::cout << "Extracted curve :" << std::endl;
			branches[i].curve.print();
			std::cout << "Expected result :" << std::endl;
			ref_branches[i].print();
		}
		ASSERT_TRUE(equal);
	}
}

TEST(TestLineDetection, cut_curve_angles) {
	std::vector<arma::fmat> branches = get_testref_savgol();
	std::vector<std::vector<arma::fmat>> ref_cuts = get_testref_cut_angles();

	int ref_i = 0;
	for (int i = 0; i < branches.size(); i++) {
		if (branches[i].n_cols == 0)
			continue;
		
		DiscreteCurve curve(branches[i]);
		std::vector<DiscreteCurve> cuts = cut_curve_angles(curve, _min_branch_length, _max_curvature);

		EXPECT_EQ(ref_cuts[ref_i].size(), cuts.size());

		for (int j = 0; j < cuts.size(); j++) {
			if (j >= ref_cuts[ref_i].size())
				continue;
			
			bool equal = arma::approx_equal(cuts[j].curve, ref_cuts[ref_i][j], "absdiff", 0.001);
			if (!equal) {
				std::cout << "Initial curve :" << std::endl;
				branches[i].print();
				std::cout << "Cut " << j << " :" << std::endl;
				cuts[j].curve.print();
				std::cout << "Reference cut " << j << " :" << std::endl;
				ref_cuts[ref_i][j].print();
			}
			ASSERT_TRUE(equal);
		}
		ref_i += 1;
	}
}

TEST(TestLineDetection, filter_lines) {
	std::vector<arma::fmat> branches = get_testref_branches();
	std::vector<DiscreteCurve> curves;
	for (auto it = branches.begin(); it != branches.end(); it++)
		curves.emplace_back(*it);

	std::vector<arma::fmat> ref_filtered = get_testref_filter_lines();
	std::vector<DiscreteCurve> filtered = filter_lines(curves, _scale_factor);

	EXPECT_EQ(filtered.size(), ref_filtered.size());

	for (int i = 0; i < filtered.size(); i++) {
		if (i >= ref_filtered.size())
			continue;

		EXPECT_EQ(filtered[i].curve.n_cols, ref_filtered[i].n_cols);

		bool equal = arma::approx_equal(filtered[i].curve, ref_filtered[i], "absdiff", 0.001);
		if (!equal) {
			filtered[i].curve.print();
			ref_filtered[i].print();
		}
		ASSERT_TRUE(equal);
	}
}
