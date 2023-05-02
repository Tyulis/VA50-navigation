#!/usr/bin/env python3
import sys
import math

import yaml
import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Float32, String, UInt8
from geometry_msgs.msg import TwistStamped

from trajectory.msg import Trajectory
from transformtrack.msg import TimeBatch
from transformtrack.srv import TransformBatch, TransformBatchRequest
from trafficsigns.msg import TrafficSignStatus, TrafficSign


RATE = 10 # Hz

class PurePursuitController (object):
    def __init__(self, parameters):

        self.parameters = parameters

        self.k = self.parameters["control"]["k"]  # look forward gain
        self.Lfc = self.parameters["control"]["Lfc"]  # [m] look-ahead distance
        self.Kp = self.parameters["control"]["Kp"]  # speed proportional gain
        self.dt = self.parameters["control"]["dt"]  # [s] time tick
        self.WB = self.parameters["control"]["WB"]  # [m] wheel base of vehicle
        self.target_speed = self.parameters["control"]["target-speed"] / 3.6  # [m/s]

        self.velocity_topic = self.parameters["node"]["velocity-topic"]
        self.trajectory_topic = self.parameters["node"]["trajectory-topic"]
        self.speed_topic = self.parameters["node"]["speed-topic"]
        self.speed_cap_topic = self.parameters["node"]["speed-cap-topic"]
        self.steering_angle_topic = self.parameters["node"]["steering-angle-topic"]
        self.traffic_sign_topic = self.parameters["node"]["traffic-sign-topic"]
        self.direction_topic = self.parameters["node"]["direction-topic"]

        # Speeds obtained from the velocity topic
        self.real_speed = None
        self.real_angular_speed = None

        # State of the controller
        self.state = None
        self.target_course = None
        self.target_ind = None
        self.is_stop_need = False
        self.stop_type = None
        self.speeds_to_stop = None
        self.current_stop_index = None
        self.is_trajectory_ready = False

        # Conserve all the states in memory
        # self.states = States()

        # Initialize the service connections
        rospy.loginfo("Waiting for the TransformBatch service...")
        self.transform_service = None
        rospy.wait_for_service(self.parameters["node"]["transform-service-name"])
        self.transform_service = rospy.ServiceProxy(self.parameters["node"]["transform-service-name"], TransformBatch, persistent=True)

        # Initialize the topic subscribers
        self.velocity_subscriber = rospy.Subscriber(self.velocity_topic, TwistStamped, self.callback_velocity)
        self.trajectory_subscriber = rospy.Subscriber(self.trajectory_topic, Trajectory, self.callback_trajectory)
        self.traffic_sign_subscriber = rospy.Subscriber(self.traffic_sign_topic, TrafficSignStatus, self.callback_traffic_sign)

        # Initialize the topic publishers
        self.speed_publisher = rospy.Publisher(self.speed_topic, Float32, queue_size=10)
        self.steering_angle_publisher = rospy.Publisher(self.steering_angle_topic, Float32, queue_size=10)
        self.direction_publisher = rospy.Publisher(self.direction_topic, UInt8, queue_size=1)
        self.speed_cap_publisher = rospy.Publisher(self.speed_cap_topic, Float32, queue_size=10)

        rospy.loginfo("Everything ready")


    def get_map_transforms(self, start_times, end_time):
        """Get the transform matrix to transform 3d homogeneous coordinates in the right target timestamp"""
        request = TransformBatchRequest()
        request.timestamps = TimeBatch(start_times=start_times, end_time=end_time)
        request.unbias = False
        tries = 0
        while True:
            try:
                response = self.transform_service(request)
                break
            except rospy.ServiceException as exc:
                if tries > 10:
                    rospy.logerr(f"Connection to service {self.parameters['node']['transform-service-name']} failed {tries} times, skipping")
                    rospy.logerr(f"Failed with error {exc}")
                    raise RuntimeError("Unable to connect to the transform service")
                rospy.logerr(f"Connection to service {self.parameters['node']['transform-service-name']} lost, reconnecting...")
                self.transform_service.close()
                self.transform_service = rospy.ServiceProxy(self.parameters["node"]["transform-service-name"], TransformBatch, persistent=True)
                tries += 1
        transforms = np.asarray(response.transforms.data).reshape(response.transforms.layout.dim[0].size, response.transforms.layout.dim[1].size, response.transforms.layout.dim[2].size).transpose(0, 2, 1)
        start_times_unbiased = start_times  #response.timestamps.start_times
        end_time_unbiased = end_time  #response.timestamps.end_time
        return transforms, start_times_unbiased, end_time_unbiased


    def callback_velocity(self, data):
        """Callback called to get the real speed and angular speed from the velocity topic"""
        #rospy.loginfo("Received a velocity")
        velocity_msg = data.twist
        velocity_x = velocity_msg.linear.x
        velocity_y = velocity_msg.linear.y
        self.real_speed = np.linalg.norm([velocity_x, velocity_y])
        self.real_angular_speed = velocity_msg.angular.z


    def callback_trajectory(self, data):
        """Callback called when a new trajectory to follow is received by the trajectory node"""
        rospy.loginfo("Received a new trajectory")

        if self.real_speed != None and self.real_angular_speed != None:

            trajectory_stamp = data.header.stamp

            trajectory_data = np.array(data.trajectory.data)
            trajectory_data = np.reshape(trajectory_data, (-1, 2))

            # Get transform matrix and apply it to the trajectory points
            transforms, _, _ = self.get_map_transforms([trajectory_stamp], rospy.get_rostime())
            transform = transforms[0]
            nb_points = trajectory_data.shape[0]
            trajectory_data_3d = np.concatenate((trajectory_data, np.zeros((nb_points, 1)), np.ones((nb_points, 1))), axis=1)
            current_trajectory_data_3d = transform @ trajectory_data_3d.T
            current_trajectory_data_3d = current_trajectory_data_3d.T
            current_trajectory_data_2d = current_trajectory_data_3d[:,:2]

            self.target_course = TargetCourse(current_trajectory_data_2d[:,1], current_trajectory_data_2d[:,0], parameters=self.parameters)

            self.state = State(parameters=self.parameters)
            self.state.update(self.real_speed, self.real_angular_speed)
            self.target_ind, _ = self.target_course.search_target_index(self.state)

            self.is_trajectory_ready = True
            rospy.loginfo("Trajectory is ready")


    def callback_traffic_sign(self, data):
        """Callback called when a new traffic sign is detected and received in the traffic sign topic"""
        rospy.loginfo("Received a new traffic sign")

        for sign in data.traffic_signs:
            # Filter out traffic signs where a stop is not needed (we chosed to stop the car only for Yields and Stops)
            if sign.type in ('stop', 'yield', 'no-entry', 'light-red', 'light-orange'):
                position = np.c_[[sign.x, sign.y, sign.z, 1]]
                transforms, _, _ = self.get_map_transforms([data.header.stamp], rospy.get_rostime())
                current_position = transforms[0] @ position
                distance = np.linalg.norm(current_position[:2])

                self.is_stop_need = True
                self.stop_type = 'light' if sign.type in ('light-red', 'light-orange') else 'sign'
                nb_speed_values_to_stop = int((distance / self.real_speed) * RATE)
                self.speeds_to_stop = np.linspace(start = self.real_speed, stop = 0, num = nb_speed_values_to_stop)
                self.current_stop_index = 0

            # Filter traffic signs where a direction is mandatory and publish on the direction topic
            elif sign.type in ('right-only', 'keep-right'):
                self.direction_publisher.publish(0b0100)
            elif sign.type in ('left-only', 'keep-left'):
                self.direction_publisher.publish(0b0010)
            elif sign.type == 'ahead-only':
                self.direction_publisher.publish(0b0001)
            elif sign.type == 'light-green' and self.is_stop_need and self.stop_type == 'light':
                self.is_stop_need = False


    def plot_arrow(self, x, y, yaw, length=1.0, width=0.5, fc="r", ec="self.k"):
        """Plot arrow (angular speed) on the graph"""
        if not isinstance(x, float):
            for ix, iy, iyaw in zip(x, y, yaw):
                self.plot_arrow(ix, iy, iyaw)
        else:
            plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                    fc=fc, ec=ec, head_width=width, head_length=width)
            plt.plot(x, y)

    
    def publish_control_inputs(self):
        if self.is_trajectory_ready:
            if self.target_ind >= len(self.target_course.cx)-1 :
                self.speed_publisher.publish(0)  # Stop the vehicule
            else:
                # Calc control input
                vi = self.speed_control()
                di, self.target_ind = self.pure_pursuit_steer_control(self.target_ind)
                print(f"vi={vi}, self.real_speed={self.real_speed}")

                self.speed_publisher.publish(vi)  # Control speed
                self.speed_cap_publisher.publish(self.target_speed)
                self.steering_angle_publisher.publish(di*180/math.pi) # Control steering angle

                self.state.update(self.real_speed, self.real_angular_speed)

                # self.states.append(self.state)

                # plt.cla()
                # # for stopping simulation with the esc key.
                # plt.gcf().canvas.mpl_connect(
                #     'key_release_event',
                #     lambda event: [exit(0) if event.key == 'escape' else None])
                # self.plot_arrow(self.state.x, self.state.y, self.state.yaw)
                # plt.plot(self.target_course.cx, self.target_course.cy, "-r", label="course")
                # plt.plot(self.states.x, self.states.y, "-b", label="trajectory")
                # plt.plot(self.target_course.cx[self.target_ind], self.target_course.cy[self.target_ind], "xg", label="target")
                # plt.axis("equal")
                # plt.grid(True)
                # plt.title("Speed[km/h]:" + str(self.state.v * 3.6)[:4])
                # plt.pause(0.001)

                rospy.loginfo("Published control inputs")


    def speed_control(self):
        """Control the speed of the car and adapt speed when stop is needed"""
        if self.is_stop_need:
            print('STOP')
            if self.current_stop_index < len(self.speeds_to_stop):
                vi = self.speeds_to_stop[self.current_stop_index]
                self.current_stop_index += 1
            elif self.stop_type == "sign":
                vi = 0
                rospy.sleep(2)
                self.is_stop_need = False
            else:
                vi = 0
        else:
            vi = self.state.v + self.Kp * (self.target_speed - self.state.v)

        return vi

    def pure_pursuit_steer_control(self, pind):
        """Control the steering angle of the car"""
        ind, Lf = self.target_course.search_target_index(self.state)

        if pind >= ind:
            ind = pind

        if ind < len(self.target_course.cx):
            tx = self.target_course.cx[ind]
            ty = self.target_course.cy[ind]
        else:  # Toward goal
            tx = self.target_course.cx[-1]
            ty = self.target_course.cy[-1]
            ind = len(self.target_course.cx) - 1

        alpha = math.atan2(ty - self.state.rear_y, tx - self.state.rear_x) - self.state.yaw

        delta = math.atan2(2.0 * self.WB * math.sin(alpha) / Lf, 1.0)

        return delta, ind


class State:

    def __init__(self, parameters, x=0.0, y=0.0, yaw=0.0, v=0.0):
        # Parameters
        self.k = parameters["control"]["k"]  # look forward gain
        self.Lfc = parameters["control"]["Lfc"]  # [m] look-ahead distance
        self.Kp = parameters["control"]["Kp"]  # speed proportional gain
        self.dt = parameters["control"]["dt"]  # [s] time tick
        self.WB = parameters["control"]["WB"]  # [m] wheel base of vehicle

        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((self.WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WB / 2) * math.sin(self.yaw))

    def update(self, v, delta):
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.v / self.WB * math.tan(delta) * self.dt
        self.v = v
        self.rear_x = self.x - ((self.WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []

    def append(self, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)


class TargetCourse:

    def __init__(self, cx, cy, parameters):
        # Parameters
        self.k = parameters["control"]["k"]  # look forward gain
        self.Lfc = parameters["control"]["Lfc"]  # [m] look-ahead distance

        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # Search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        # Update look ahead distance
        Lf = self.k * state.v + self.Lfc

        # Search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # Not exceed goal
            ind += 1

        return ind, Lf


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <parameter-file>")
    else:
        with open(sys.argv[1], "r") as parameterfile:
            parameters = yaml.load(parameterfile, yaml.Loader)

        rospy.init_node("pure_pursuit_control")
        node = PurePursuitController(parameters)
        while not rospy.is_shutdown():
            rate = rospy.Rate(RATE)
            node.publish_control_inputs()
            rate.sleep()

        rospy.spin()
