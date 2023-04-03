#include "transformtrack/TransformTrackNode.h"

int main(int argc, char** argv) {
	ros::init(argc, argv, config::node::transform_node_name);

	TransformTrackNode node;
	node.run();
}