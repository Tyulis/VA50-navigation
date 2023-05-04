#!/usr/bin/env python3

#   Copyright 2023 Grégori MIGNEROT, Élian BELMONTE, Benjamin STACH
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import sys
from threading import Lock

import yaml
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import rospy
from visualization.msg import VizUpdate


class VisualizationNode (object):
	def __init__(self, parameters):
		self.parameters = parameters
		self.update_subscriber = rospy.Subscriber(self.parameters["node"]["visualization-topic"], VizUpdate, self.callback_update, queue_size=10, buff_size=2**28)
		self.viz_image = 255 * np.ones((1080, 1920, 3), dtype=np.uint8)

		self.rects = {
			self.parameters["visualization"]["circulation-lines-id"]     : [  0,   0,  950, 538],
			self.parameters["visualization"]["circulation-trajectory-id"]: [970,   0,  950, 538],
			self.parameters["visualization"]["trafficsigns-id"]          : [  0, 558, 1920, 522],
		}

		#cv.namedWindow("viz", cv.WINDOW_NORMAL)
		#cv.resizeWindow("viz", self.viz_image.shape[1], self.viz_image.shape[0])
		#cv.imshow("viz", self.viz_image)
		plt.imshow(self.viz_image)
		rospy.loginfo("Ready")

	def callback_update(self, message):
		image = np.frombuffer(message.image.data, dtype=np.uint8).reshape((message.image.height, message.image.width, 3))
		if message.id == self.parameters["visualization"]["trafficsigns-id"]:
			image = image[200:600]
		rect = self.rects[message.id]
		image = cv.resize(image, (rect[2], rect[3]))
		self.viz_image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = image
		plt.imshow(self.viz_image)
		plt.draw()


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <parameter-file>")
	else:
		# Load the parameters and map
		with open(sys.argv[1], "r") as parameterfile:
			parameters = yaml.load(parameterfile, yaml.Loader)

		# Initialize and start the node
		rospy.init_node(parameters["node"]["visualization-node-name"])
		node = VisualizationNode(parameters)
		plt.show(block=True)
