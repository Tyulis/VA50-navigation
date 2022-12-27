#!/usr/bin/env python3
import rospy

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from sensor_msgs.msg import Image

#Homography Image Parameter
MID_POINT = 514
BOT_LIM = 700
TOP_LIM = 450
BOTTOM_LEFT_LIM = 60
BOTTOM_RIGHT_LIM = MID_POINT*2-60
TOP_LEFT_LIM = MID_POINT-80
TOP_RIGHT_LIM = MID_POINT+80
REAL_VIEW = np.array([(BOTTOM_LEFT_LIM, BOT_LIM), (BOTTOM_RIGHT_LIM, BOT_LIM) ,(TOP_RIGHT_LIM, TOP_LIM), (TOP_LEFT_LIM, TOP_LIM)])

#Homography birdeye view parameters
BV_HEIGHT = 500
BV_WIDTH = 440
BIRD_VIEW = np.array([(0, BV_HEIGHT), (BV_WIDTH, BV_HEIGHT), (BV_WIDTH, 0), (0, 0)])

#Homography
HOMO, _ = cv.findHomography(REAL_VIEW, BIRD_VIEW)
HOMO_INV = np.linalg.inv(HOMO)

#Filters
HORIZONTAL_FILTER = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])


class LaneDetector(object):
    def __init__(self, image_topic):
        self.image_topic = image_topic

        # Initialize the topic subscribers
        self.image_subscriber = rospy.Subscriber(self.image_topic, Image, self.callback_image)

        # Initialize the publishers
        self.image_pub = rospy.Publisher('laneDetection', Image, queue_size=10)

        # Initialize image bridge
        self.bridge = CvBridge()

        self.image = None
        self.detection_image = None

        #Current lane detected
        self.leftline = None
        self.rightline = None
        self.midlane = None
        self.midpts = None

        #Previous lane detected
        self.last_leftline = None
        self.last_rightline = None
        self.last_midlane = None

        rospy.loginfo("Everything ready")


    def callback_image(self, data):
        """Extract an image from the camera"""
        rospy.loginfo("Received an image")
        self.image_frame = data.header.frame_id
        self.image = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, 3))

        #Initialize previous lanes
        self.last_leftline = self.leftline
        self.last_rightline = self.rightline
        self.last_midlane = self.midlane
        #Reinitialize current lanes
        self.leftline = None
        self.rightline = None
        self.midlane = None

        self.process_image()


    def bird_view(self, img, height, width, reverse=False):
        """Perform a birdeye view transformation (or reverse)"""
        if not reverse:
            new_img = cv.warpPerspective(img, HOMO, (width, height), cv.INTER_LINEAR)
        else:
            new_img = cv.warpPerspective(img, HOMO_INV , (width, height), cv.INTER_LINEAR)
        return new_img


    def box_points(self, rect, all_pts, rect_base, offset):
        """ 1. Append the pixels in the counting box to the lane list
            2. Calculate the x base of the next counting box (mean of x axis)
            3. Count the number of pixel in the counting box"""
        rect_pts = np.argwhere(rect==255)
        ly = np.unique(rect_pts.T[0])
        lx = []
        for y in ly:
            mean = []
            for coor in rect_pts:
                if y == coor[0]:
                    mean.append(coor[1])
            mean = np.mean(np.array(mean))
            lx.append(mean)
        lx = np.array(lx).astype(int)
        if rect_pts.size != 0:
            lx = lx+offset[0]
            ly = ly+offset[1]
            all_pts.append(np.array([ly, lx]))
            return int(np.mean(lx)), all_pts, lx.size
        else:
            return rect_base, all_pts, 0


    def detect_lines(self, img):
        detect_img = img.copy()
        detect_img = cv.merge([detect_img, detect_img, detect_img])

        height = img.shape[0]
        width = img.shape[1]

        left_pts = []
        right_pts = []
        l_continue, r_continue = True, True

        #Find lane bases (with histogram)
        hist = np.sum(img[img.shape[0]//2:,:], axis=0)
        mid_point = img.shape[1]//2
        leftx_base = np.argmax(hist[:mid_point])
        rightx_base = np.argmax(hist[mid_point:]) + mid_point

        #Creating windows to isolate lanes
        rect_bottom = height
        rect_height = 50
        rect_offset = 60
        while rect_bottom > 0:
            #Set windows limits
            x1_left = leftx_base - min(leftx_base, rect_offset)
            x2_left = leftx_base + rect_offset
            x1_right = rightx_base - rect_offset
            x2_right = rightx_base + min(width-rightx_base, rect_offset)
            y1 = rect_bottom-rect_height
            y2 = rect_bottom

            if l_continue:
                left_rect = img[y1:y2, x1_left:x2_left]
                #Color lanes pixels
                detect_img[y1:y2, x1_left:x2_left] = cv.merge([left_rect, np.zeros_like(left_rect), np.zeros_like(left_rect)])
                detect_img = cv.rectangle(detect_img, (x1_left,y1), (x2_left, y2), (0,255,0), 1)
                leftx_base, left_pts, lnew_pts = self.box_points(left_rect, left_pts, leftx_base, (x1_left, y1))
                if lnew_pts == 0 and leftx_base<rect_offset:
                    l_continue = False

            if r_continue:
                right_rect = img[y1:y2, x1_right:x2_right]
                #Color lanes pixels
                detect_img[y1:y2, x1_right:x2_right] = cv.merge([np.zeros_like(right_rect), np.zeros_like(right_rect), right_rect])
                detect_img = cv.rectangle(detect_img, (x1_right,y1), (x2_right, y2), (0,255,0), 1)
                rightx_base, right_pts, rnew_pts  = self.box_points(right_rect, right_pts, rightx_base, (x1_right, y1))
                if rnew_pts == 0 and width-rightx_base<rect_offset:
                    r_continue = False

            rect_bottom -= rect_height

        #Compute curves
        try:
            left_pts = np.concatenate(left_pts, axis=1)
            self.leftline = np.polyfit(left_pts[0], left_pts[1], 2)
        except ValueError:
            rospy.loginfo("Left line not detected.")

        try:
            right_pts = np.concatenate(right_pts, axis=1)
            self.rightline = np.polyfit(right_pts[0],right_pts[1], 2)
        except ValueError:
            rospy.loginfo("Right line not detected.")

        return detect_img


    def compute_lane(self, p):
        """Compute lanes with polynomial function (p)"""
        x = np.linspace(0, BV_HEIGHT, 300).astype(int)
        y = np.polyval(p, x)
        y = y.astype(int)
        poly = np.array([y, x]).T
        return poly


    def bird_to_image(self, pts):
        """Convert birdeye view point coordinates to image"""
        pts = np.asarray(pts).T
        pts = np.concatenate([pts, np.array([[1]*pts.shape[1]])])
        new_pts = HOMO_INV @ pts
        new_pts = new_pts/new_pts[2]
        return new_pts


    def detect_road(self, img):
        """ 1. Determine the center lane to follow based on the right and left line
            2. Draw the road and the detected lines."""
        detect_img = np.zeros_like(img)

        #Compute points of the detected lines
        if self.leftline is not None:
            lline = self.compute_lane(self.leftline)
        if self.rightline is not None:
            rline = self.compute_lane(self.rightline)
            rline = np.flip(rline, axis=0)

        #Compute center of the lane based on detected lines
        if self.leftline is not None and self.leftline is not None:
            self.midlane = (self.leftline+self.rightline)/2
            mp = self.compute_lane(self.midlane)
            detect_img = cv.fillPoly(detect_img, [np.concatenate([lline, rline])] ,(0,255,0))
            detect_img = cv.polylines(detect_img, [mp], False, (0,0,255), 5)
        if self.leftline is not None:
            mp = self.leftline
            print(mp[2])
            mp[2] = mp[2]+(BV_WIDTH-lline[-1][0])//2
            detect_img = cv.polylines(detect_img, [mp], False, (255,0,0), 3)
        if self.rightline is not None:
            mp = self.rightline
            print(mp[2])
            mp[2] = mp[2]+(BV_WIDTH*(1-rline[-1][0]))//2
            detect_img = cv.polylines(detect_img, [mp], False, (0,255,0), 1)
        
        self.midpts = self.bird_to_image(mp)

        try:
            detect_img = cv.polylines(detect_img, [rline], False, (255,0,255), 5)
        except:
            pass
        try:     
            detect_img = cv.polylines(detect_img, [lline], False, (255,0,255), 5)
        except:
            pass

        return detect_img


    def add_views(self, img, views, center=False):
        """Append different views (of the pipeline processing) the image"""
        views = [cv.merge([v,v,v]) if (len(v.shape) == 2) else v for v in views]
        view = cv.hconcat(views)
        height = view.shape[0]
        width = view.shape[1]
        if center:
            x_pos = MID_POINT-width//2
        else:
            x_pos = 0
        y_pos = 0
        view = cv.resize(view, (self.image.shape[1], view.shape[0]))
        img[y_pos:y_pos+height, x_pos:x_pos+width] = view
        return img


    def process_image(self):
        "Processing pipeline"
        views = []
        #All views
        bv_image = None
        gs_image = None
        bin_image = None
        lane_image = None
        road_image = None

        #Image processing
        #Image to birdview
        bv_image = self.bird_view(self.image, BV_HEIGHT, BV_WIDTH)
        #Image to grayscale
        gs_image = cv.cvtColor(bv_image, cv.COLOR_RGB2GRAY)
        #Image to binary
        bin_image = cv.GaussianBlur(gs_image, (5,5), 0)
        bin_image  = cv.inRange(bin_image , 128, 255)
        #Find lane
        lane_image = self.detect_lines(bin_image)
        #Display road
        road_image = self.detect_road(lane_image)

        #Birdview to image
        roadmask_img = self.bird_view(road_image, self.image.shape[0], self.image.shape[1], reverse=True)
        self.detection_image = cv.addWeighted(self.image, 1, roadmask_img, 0.3, 1)

        views.append(cv.resize(bv_image, (220, 250)))
        views.append(cv.resize(gs_image, (220, 250)))
        views.append(cv.resize(bin_image, (220, 250)))
        views.append(cv.resize(lane_image, (220, 250)))
        views.append(cv.resize(road_image, (220, 250)))
        self.detection_image = self.add_views(self.detection_image , views)

        #Publish detection image (mainly for testing)
        self.publish_image(self.detection_image)


    def publish_image(self, img):
        """Publish detection image (mainly for testing)"""
        msg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        self.image_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("lane_detection")
    node = LaneDetector("/forwardCamera/image_raw")
    rospy.spin()