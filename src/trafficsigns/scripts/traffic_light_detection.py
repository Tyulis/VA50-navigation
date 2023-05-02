import cvlib
import cv2 as cv
import numpy as np
from trafficsign import TrafficSign



def detect_traffic_lights(image):
	boxes, labels, confidences = cvlib.detect_common_objects(image)

	signs = []
	for bbox, label, confidence in zip(boxes, labels, confidences):
		if label != "traffic light":
			continue

		# Now check for the current lit color
		area = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		hue = cv.cvtColor(area, cv.COLOR_RGB2HSV)[:, :, 0]

		# Take the hue shifted by 180° (90 for OpenCV)
		# This prevents red from getting split between 0 and 180, and puts blue (= sky = noise) in its place
		# Then only keep the relevant parts (red-orange-green ≈ -10 – 70° ⟶ 80–160 here) of the histogram
		hue_hist = cv.calcHist([(hue + 90) % 180], [0], None, [20], [80, 160])
		red = np.sum(hue_hist[0:5])
		orange = np.sum(hue_hist[5:10])
		green = np.sum(hue_hist[14:19])
		total = red + orange + green
		red /= total
		orange /= total
		green /= total

		print(red, orange, green)

		if red > 0.4: type = "light-red"
		elif green > 0.4: type = "light-green"
		elif orange > 0.4: type = "light-orange"
		else: continue
		
		sign = TrafficSign("trafficlight", type, f"Traffic light, {type}", bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], confidence)
		signs.append(sign)

		left, top, right, bottom = bbox
		global_label = str(sign.label) + "=" + str(round(confidence*100, 2)) + "%"
				
		cv.rectangle(image, (left, top), (right, bottom), (0,0,255), 2)
		cv.putText(image, global_label, (left, bottom+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
	return image, signs
