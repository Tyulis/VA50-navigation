#include <iostream>
#include <gtest/gtest.h>

#include "trajectory/DiscreteCurve.h"
#include "trajectory/DiscreteGeometry.h"

#include "testref_resample.h"
#include "testref_savgol.h"
#include "testref_branches.h"
#include "testref_length.h"
#include "testref_filtered_target.h"
#include "testref_gradient.h"
#include "testref_project_point.h"
#include "testref_vector_angle.h"
#include "testref_dilate.h"
#include "testref_curvature.h"

TEST(TestDiscreteCurve, length) {
	std::vector<arma::fmat> branches = get_testref_branches();
	std::vector<float> lengths = get_testref_length();

	for (int i = 0; i < branches.size(); i++) {
		DiscreteCurve curve(branches[i]);
		float length = curve.length();
		if (abs(length - lengths[i]) > 0.0001) {
			std::cout << "Initial curve " << i << " :" << std::endl;
			branches[i].print();
			std::cout << "Expected " << length << ", found " << lengths[i] << std::endl;
		}
		ASSERT_LE(abs(length - lengths[i]), 0.0001);
	}
}

TEST(TestDiscreteCurve, resample) {
	std::vector<arma::fmat> branches = get_testref_branches();
	std::vector<arma::fmat> resampled = get_testref_resample();

	for (int i = 0; i < branches.size(); i++) {
		DiscreteCurve curve(branches[i]);
		curve.resample(_resample_step);
		bool equal = arma::approx_equal(curve.curve, resampled[i], "absdiff", 0.001);
		if (!equal) {
			std::cout << "Initial curve [" << i << "] :" << std::endl;
			branches[i].print();
			std::cout << "Resampled curve :" << std::endl;
			curve.curve.print();
			std::cout << "Expected result :" << std::endl;
			resampled[i].print();
		}
		ASSERT_TRUE(equal);
	}
}

TEST(TestDiscreteCurve, savgol) {
	std::vector<arma::fmat> branches = get_testref_resample();
	std::vector<arma::fmat> savgol = get_testref_savgol();

	for (int i = 0; i < branches.size(); i++) {
		if (branches[i].n_cols <= 3)
			continue;
		DiscreteCurve curve(branches[i]);
		curve.savgol_filter(_savgol_window);
		bool equal = arma::approx_equal(curve.curve, savgol[i], "absdiff", 0.001);
		if (!equal) {
			std::cout << "Initial curve [" << i << "] :" << std::endl;
			branches[i].print();
			std::cout << "Filtered curve :" << std::endl;
			curve.curve.print();
			std::cout << "Expected result :" << std::endl;
			savgol[i].print();
		}
		ASSERT_TRUE(equal);
	}
}

TEST(TestDiscreteCurve, gradient) {
	std::vector<arma::fmat> branches = get_testref_filtered_target();
	std::vector<arma::fmat> ref_gradient = get_testref_gradient();

	for (int i = 0; i < branches.size(); i++) {
		DiscreteCurve curve(branches[i]);
		arma::fmat gradient = curve.gradient();
		bool equal = arma::approx_equal(gradient, ref_gradient[i], "absdiff", 0.001);
		if (!equal) {
			std::cout << "Initial curve [" << i << "] :" << std::endl;
			branches[i].print();
			std::cout << "Gradient :" << std::endl;
			gradient.print();
			std::cout << "Expected gradient :" << std::endl;
			ref_gradient[i].print();
		}
		ASSERT_TRUE(equal);
	}
}

TEST(TestDiscreteCurve, project_point) {
	auto [projection_source, projection_dest_mat] = get_testref_projection_curves();
	DiscreteCurve projection_dest(projection_dest_mat);
	std::vector<std::tuple<int, float>> ref_projections = get_testref_projection_results();

	for (int i = 0; i < projection_source.n_cols; i++) {
		auto [ref_index, ref_segment_part] = ref_projections[i];
		auto [index, segment_part] = projection_dest.project_point(projection_source.col(i), false);
		ASSERT_EQ(index, ref_index);
		ASSERT_NEAR(segment_part, ref_segment_part, 0.0001);
	}
}

TEST(TestDiscreteCurve, vector_angle) {
	arma::fmat curve = get_testref_vector_angle_curve();
	std::vector<float> ref_angles = get_testref_vector_angle();

	for (int i = 0; i < curve.n_cols - 1; i++) {
		float angle = vector_angle(curve.col(i), curve.col(i+1));
		ASSERT_NEAR(angle, ref_angles[i], 0.0001);
	}
}

TEST(TestDiscreteCurve, dilate_left) {
	std::vector<arma::fmat> branches = get_testref_filtered_target();
	std::vector<DiscreteCurve> curves;
	for (auto it = branches.begin(); it != branches.end(); it++)
		curves.emplace_back(*it);
	
	std::vector<arma::fmat> ref_dilated = get_testref_dilate_left();
	for (int i = 0; i < curves.size(); i++) {
		DiscreteCurve curve = curves[i];
		curve.dilate(_dilation, _left_direction);
		bool equal = arma::approx_equal(curve.curve, ref_dilated[i], "absdiff", 0.001);
		if (!equal) {
			std::cout << "Initial curve :" << std::endl;
			branches[i].print();
			std::cout << "Dilated :" << std::endl;
			curve.curve.print();
			std::cout << "Ref dilated :" << std::endl;
			ref_dilated[i].print();
		}
		ASSERT_TRUE(equal);
	}
}

TEST(TestDiscreteCurve, dilate_right) {
	std::vector<arma::fmat> branches = get_testref_filtered_target();
	std::vector<DiscreteCurve> curves;
	for (auto it = branches.begin(); it != branches.end(); it++)
		curves.emplace_back(*it);
	
	std::vector<arma::fmat> ref_dilated = get_testref_dilate_right();
	for (int i = 0; i < curves.size(); i++) {
		DiscreteCurve curve = curves[i];
		curve.dilate(_dilation, _right_direction);
		bool equal = arma::approx_equal(curve.curve, ref_dilated[i], "absdiff", 0.001);
		if (!equal) {
			std::cout << "Initial curve :" << std::endl;
			branches[i].print();
			std::cout << "Dilated :" << std::endl;
			curve.curve.print();
			std::cout << "Ref dilated :" << std::endl;
			ref_dilated[i].print();
		}
		ASSERT_TRUE(equal);
	}
}

TEST(TestDiscreteCurve, mean_curvature) {
	std::vector<arma::fmat> branches = get_testref_filtered_target();
	std::vector<float> ref_curvature = get_testref_mean_curvature();

	for (int i = 0; i < branches.size(); i++) {
		float curvature = mean_curvature(branches[i]);
		ASSERT_NEAR(curvature, ref_curvature[i], 0.0001);
	}
}