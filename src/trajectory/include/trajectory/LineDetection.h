#ifndef _TRAJECTORY_LINEDETECTION_H
#define _TRAJECTORY_LINEDETECTION_H

#include <vector>
#include <armadillo>
#include <opencv2/opencv.hpp>

#include "trajectory/DiscreteCurve.h"


struct MergeCandidate {
	bool merge;         // True => the lines can be merged, False => they don’t
	bool flip1;         // Whether to flip the first line before concatenation
	bool flip2;         // Whether to flip the second line before concatenation
	bool arc;           // Whether the join is a circle arc
	arma::fvec center;  // Circle arc center
	float radius;       // Circle arc radius
	float error;        // Mean squared error of the merge
	float distance;     // Minimal distance between the lines

	inline MergeCandidate() : merge(false), flip1(false), flip2(false), arc(false), radius(0), error(-1), distance(0) {}
};


std::vector<DiscreteCurve> extract_branches(cv::Mat const& be_binary);
std::vector<DiscreteCurve> filter_lines(std::vector<DiscreteCurve> const& branches, float scale_factor);
std::vector<DiscreteCurve> cut_curve_angles(DiscreteCurve const& curve, float min_length, float max_curvature);
MergeCandidate check_mergeability(DiscreteCurve const& line1, DiscreteCurve const& line2);
void fit_line(MergeCandidate& result, DiscreteCurve const& line1, DiscreteCurve const& line2, int start1, int end1, int start2, int end2);
void fit_arc_kasa(MergeCandidate& result, DiscreteCurve const& line1, DiscreteCurve const& line2, int start1, int end1, int start2, int end2);
DiscreteCurve join_curves_line(MergeCandidate candidate, DiscreteCurve line1, DiscreteCurve line2);
DiscreteCurve join_curves_arc(MergeCandidate candidate, DiscreteCurve line1, DiscreteCurve line2, float branch_step);


#endif