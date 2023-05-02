#include "ros/ros.h"

#include "config.h"
#include "trajectory/TrajectoryExtractorNode.h"

void print_arma(arma::fmat matrix) {
	matrix.print();
}
void print_arma(arma::umat matrix) {
	matrix.print();
}
void print_arma(arma::fvec matrix) {
	matrix.print();
}
void print_arma(arma::uvec matrix) {
	matrix.print();
}
void print_arma(arma::frowvec matrix) {
	matrix.print();
}
void print_arma(arma::urowvec matrix) {
	matrix.print();
}

int main(int argc, char** argv) {
	ros::init(argc, argv, config::node::trajectory_node_name);

	TrajectoryExtractorNode node;
	node.run();
}