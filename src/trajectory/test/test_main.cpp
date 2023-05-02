#include <gtest/gtest.h>

#include "test_DiscreteCurve.hpp"
#include "test_LineDetection.hpp"
#include "test_FuzzyLanes.hpp"


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "tester");
  ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}