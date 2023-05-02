#ifndef _TRAJECTORY_INTERSECTIONHINT_H
#define _TRAJECTORY_INTERSECTIONHINT_H

#include <string>
#include <vector>
#include <armadillo>

#include "ros/ros.h"
#include "trajectory/Direction.h"


/* Hold the position and informations about anything that may indicate an intersection */
struct IntersectionHint {
	enum class Category {
		TrafficSign, Marking, 
	};


	IntersectionHint(IntersectionHint::Category category_, std::string const& type_, arma::fvec const& position, ros::Time timestamp, float confidence);

	void merge(IntersectionHint const& hint);
	float confidence() const;
	Direction direction_hint() const;

	static constexpr float match_threshold(IntersectionHint::Category category) {
		switch (category) {
			case IntersectionHint::Category::TrafficSign: return config::intersection::intersection_hint_match_threshold::trafficsign;
			case IntersectionHint::Category::Marking    : return config::intersection::intersection_hint_match_threshold::marking;
		}
	}

	IntersectionHint::Category category;
	std::string type;
	std::vector<arma::fvec> positions;
	std::vector<ros::Time> timestamps;
	std::vector<float> confidences;
};

#endif