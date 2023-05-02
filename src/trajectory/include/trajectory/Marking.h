#ifndef _TRAJECTORY_MARKINGTYPE_H
#define _TRAJECTORY_MARKINGTYPE_H

#include <vector>
#include <armadillo>

#include "trajectory/DiscreteCurve.h"

struct Marking {
	public:
		enum class Type {
			Crosswalk,
		};

		Marking::Type type;
		arma::fcube data;
		float confidence;

		inline Marking(Marking::Type type_) : type(type_), confidence(1.0f) {}
		inline Marking(Marking::Type type_, float confidence_) : type(type_), confidence(confidence_) {}
		inline Marking(Marking::Type type_, int n_rows, int n_cols, int n_slices) : type(type_), data(n_rows, n_cols, n_slices), confidence(1.0f) {}
		inline Marking(Marking::Type type_, float confidence_, int n_rows, int n_cols, int n_slices) : type(type_), confidence(confidence_), data(n_rows, n_cols, n_slices) {}
};

std::vector<Marking> detect_markings(std::vector<DiscreteCurve>& branches, float scale_factor);
void detect_crosswalks(std::vector<Marking>& markings, std::vector<DiscreteCurve>& branches, float scale_factor);

#endif