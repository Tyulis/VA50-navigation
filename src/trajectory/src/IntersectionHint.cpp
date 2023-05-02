#include "trajectory/Statistics.h"
#include "trajectory/IntersectionHint.h"


IntersectionHint::IntersectionHint(IntersectionHint::Category category_, std::string const& type_, arma::fvec const& position, ros::Time timestamp, float confidence) {
	category = category_;
	type = type_;
	positions.push_back(position);
	timestamps.push_back(timestamp);
	confidences.push_back(confidence);
}

void IntersectionHint::merge(IntersectionHint const& hint) {
	positions.insert(positions.end(), hint.positions.begin(), hint.positions.end());
	timestamps.insert(timestamps.end(), hint.timestamps.begin(), hint.timestamps.end());
	confidences.insert(confidences.end(), hint.confidences.begin(), hint.confidences.end());
}

float IntersectionHint::confidence() const {
	return confidence_combination(confidences);
}

Direction IntersectionHint::direction_hint() const {
	if (category != IntersectionHint::Category::TrafficSign)
		return Direction::All;
	
	else if (type == "right-only" || type == "keep-right")
		return Direction::Right;
	else if (type == "left-only" || type == "keep-left")
		return Direction::Left;
	else if (type == "ahead-only")
		return Direction::Forward;
	else if (type == "straight-left-only")
		return Direction::ForwardLeft;
	else if (type == "straight-right-only")
		return Direction::ForwardRight;
	
	return Direction::All;
}