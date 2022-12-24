#!/usr/bin/env python3
import os
import sys
import rospy
from sensor_msgs.msg import Image

import numpy as np
import PIL.Image

def extract_image(data):
    rospy.loginfo(f"Extracting image {data.header.stamp}")
    shape = (data.height, data.width, 3)
    array = np.asarray(tuple(data.data), dtype=np.uint8).reshape(shape)
    image = PIL.Image.fromarray(array, "RGB")
    image.save(os.path.join(sys.argv[1], f"img-{data.header.stamp}.png"))

def listener():
    rospy.init_node("extract_image", anonymous=True)
    rospy.Subscriber("frontCamera/image_raw", Image, extract_image)
    rospy.spin()

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage : extract_image.py <output path>")
	else:
		listener()
