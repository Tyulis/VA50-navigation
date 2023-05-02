#include <iostream>
#include <gtest/gtest.h>

#include "trajectory/DiscreteCurve.h"
#include "trajectory/FuzzyLaneSystem.h"

#include "testref_filtered_target.h"
#include "testref_fuzzy_parameters.h"
#include "testref_fuzzy_scores.h"

TEST(TestFuzzyLaneSystem, estimate_main_angle) {
	std::vector<arma::fmat> branches = get_testref_filtered_target();
	std::vector<DiscreteCurve> curves;
	for (auto it = branches.begin(); it != branches.end(); it++)
		curves.emplace_back(*it);
	
	float angle = estimate_main_angle(curves);
	ASSERT_NEAR(angle, _main_angle, 0.002);
}

TEST(TestFuzzyLaneSystem, fuzzy_lane_parameters) {
	std::vector<arma::fmat> branches = get_testref_filtered_target();
	std::vector<DiscreteCurve> curves;
	for (auto it = branches.begin(); it != branches.end(); it++)
		curves.emplace_back(*it);
	
	auto [ref_forward_distance, ref_left_line_distance, ref_right_line_distance,
	      ref_line_lengths, ref_parallel_distance, ref_parallel_angle] = get_testref_fuzzy_parameters();
	auto [forward_distance, left_line_distance, right_line_distance,
	      line_lengths, parallel_distance, parallel_angle] = fuzzy_lane_parameters(curves, _main_angle, _main_angle_distance);
	
	bool eq_forward_distance = arma::approx_equal(forward_distance, ref_forward_distance, "absdiff", 0.001);
	bool eq_left_line_distance = arma::approx_equal(left_line_distance, ref_left_line_distance, "absdiff", 0.001);
	bool eq_right_line_distance = arma::approx_equal(right_line_distance, ref_right_line_distance, "absdiff", 0.001);
	bool eq_line_lengths = arma::approx_equal(line_lengths, ref_line_lengths, "absdiff", 0.001);
	bool eq_parallel_distance = arma::approx_equal(parallel_distance, ref_parallel_distance, "reldiff", 0.03);
	bool eq_parallel_angles = arma::approx_equal(parallel_angle, ref_parallel_angle, "reldiff", 0.03);

	if (!eq_forward_distance) {
		std::cout << "Forward distance :" << std::endl;
		forward_distance.print();
		std::cout << "Ref forward distance :" << std::endl;
		ref_forward_distance.print();
	}
	ASSERT_TRUE(eq_forward_distance);

	if (!eq_left_line_distance) {
		std::cout << "Left line distance :" << std::endl;
		left_line_distance.print();
		std::cout << "Ref left line distance :" << std::endl;
		ref_left_line_distance.print();
	}
	ASSERT_TRUE(eq_left_line_distance);

	if (!eq_right_line_distance) {
		std::cout << "Right line distance :" << std::endl;
		right_line_distance.print();
		std::cout << "Ref right line distance :" << std::endl;
		ref_right_line_distance.print();
	}
	ASSERT_TRUE(eq_right_line_distance);

	if (!eq_line_lengths) {
		std::cout << "Line length :" << std::endl;
		line_lengths.print();
		std::cout << "Ref line length :" << std::endl;
		ref_line_lengths.print();
	}
	ASSERT_TRUE(eq_line_lengths);

	if (!eq_parallel_distance) {
		std::cout << "Parallel distance :" << std::endl;
		parallel_distance.print();
		std::cout << "Ref parallel distance :" << std::endl;
		ref_parallel_distance.print();
	}
	ASSERT_TRUE(eq_parallel_distance);

	if (!eq_parallel_angles) {
		arma::find(ref_parallel_angle - parallel_angle > 0.00001).print();
		std::cout << "Parallel angle :" << std::endl;
		parallel_angle.print();
		std::cout << "Ref parallel angle :" << std::endl;
		ref_parallel_angle.print();
	}
	ASSERT_TRUE(eq_parallel_angles);
}

TEST(TestFuzzyLaneSystem, fuzzy_scores) {
	arma::fmat centers(5, 3);
	centers.row(0) = arma::frowvec(&config::fuzzy_lines::centers::forward_distance[0], 3);
	centers.row(1) = arma::frowvec(&config::fuzzy_lines::centers::line_distance[0], 3);
	centers.row(2) = arma::frowvec(&config::fuzzy_lines::centers::line_lengths[0], 3);
	centers.row(3) = arma::frowvec(&config::fuzzy_lines::centers::parallel_distances[0], 3);
	centers.row(4) = arma::frowvec(&config::fuzzy_lines::centers::parallel_angles[0], 3);

	arma::imat malus(5, 3);
	malus.row(0) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::forward_distance[0], 3));
	malus.row(1) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::line_distance[0], 3));
	malus.row(2) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::line_lengths[0], 3));
	malus.row(3) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::parallel_distances[0], 3));
	malus.row(4) = arma::conv_to<arma::irowvec>::from(arma::Row<int>(&config::fuzzy_lines::malus::parallel_angles[0], 3));
	
	arma::fvec output_centers(&config::fuzzy_lines::centers::output[0], 5);

	FuzzyLaneSystem lane_fuzzysystem(centers, malus, output_centers, config::fuzzy_lines::base_score);

	arma::fcube variables = get_testref_fuzzy_variables();
	arma::fmat ref_scores = get_testref_fuzzy_scores();
	arma::fmat scores = lane_fuzzysystem.fuzzy_scores(variables);
	
	bool equal = arma::approx_equal(scores, ref_scores, "absdiff", 0.001);
	if (!equal) {
		std::cout << "Scores :" << std::endl;
		scores.print();
		std::cout << "Reference scores :" << std::endl;
		ref_scores.print();
	}
	ASSERT_TRUE(equal);
}