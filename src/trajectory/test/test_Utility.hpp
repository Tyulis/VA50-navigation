#include <iostream>
#include <gtest/gtest.h>

#include "trajectory/Utility.h"


TEST(TestUtility, transform_to_sXYZ_euler) {
	{
		arma::fmat transform = {{1.0f, 0.0f, 0.0f},
		                        {0.0f, 1.0f, 0.0f},
								{0.0f, 0.0f, 1.0f}};
		std::array<float, 3> euler = transform_to_sXYZ_euler(transform);
		ASSERT_NEAR(euler[0], 0.0f, 0.00001f); ASSERT_NEAR(euler[1], 0.0f, 0.00001f); ASSERT_NEAR(euler[2], 0.0f, 0.00001f);
	}
	
	{
		arma::fmat transform = {{ 0.29192658, -0.07207501,  0.95372117},
                                { 0.45464871,  0.88774982, -0.07207501},
                                {-0.84147098,  0.45464871,  0.29192658}};
		std::array<float, 3> euler = transform_to_sXYZ_euler(transform);
		ASSERT_NEAR(euler[0], 1.0f, 0.00001f); ASSERT_NEAR(euler[1], 1.0f, 0.00001f); ASSERT_NEAR(euler[2], 1.0f, 0.00001f);
	}

	{
		arma::fmat transform = {{ 0.87758256, -0.22984885, -0.42073549},
                                { 0.        ,  0.87758256, -0.47942554},
                                { 0.47942554,  0.42073549,  0.77015115}};
		std::array<float, 3> euler = transform_to_sXYZ_euler(transform);
		ASSERT_NEAR(euler[0], 0.5f, 0.00001f); ASSERT_NEAR(euler[1], -0.5f, 0.00001f); ASSERT_NEAR(euler[2], 0.0f, 0.00001f);
	}

	{
		arma::fmat transform = {{ 0.35355339, -0.8660254 , -0.35355339},
                                { 0.61237244,  0.5       , -0.61237244},
                                { 0.70710678,  0.        ,  0.70710678}};
		std::array<float, 3> euler = transform_to_sXYZ_euler(transform);
		ASSERT_NEAR(euler[0], 0.0f, 0.00001f); ASSERT_NEAR(euler[1], -M_PI/4, 0.00001f); ASSERT_NEAR(euler[2], M_PI/3, 0.00001f);
	}

	{
		arma::fmat transform = {{ 0.31532236,  0.91948299, -0.23478255},
                                {-0.94898462,  0.30551975, -0.078012  },
                                {-0.        ,  0.24740396,  0.96891242}};
		std::array<float, 3> euler = transform_to_sXYZ_euler(transform);
		ASSERT_NEAR(euler[0], 0.25f, 0.00001f); ASSERT_NEAR(euler[1], 0.0f, 0.00001f); ASSERT_NEAR(euler[2], -1.25f, 0.00001f);
	}
}