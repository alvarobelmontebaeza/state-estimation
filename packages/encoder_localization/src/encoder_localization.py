#!/usr/bin/env python3
import numpy as np
import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Pose2DStamped, Twist2DStamped, WheelEncoderStamped
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import TransformStamped

class EncoderLocalizationNode(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

        # Initialize the DTROS parent class
        super(EncoderLocalizationNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = os.environ['VEHICLE_NAME']

        # State variables for the robot
        self.pose = Pose2DStamped(0.0,0.0,0.0) # Initial state given arbitrarily
        self.twist = Twist2DStamped(0.0,0.0)
        self.last_t = rospy.get_time()

        # Transform that defines robot state
        self.current_state = TransformStamped()
        self.current_state.header.frame_id = 'map'
        self.current_state.child_frame_id = 'encoder_baselink'

        # Values computed using encoder data
        self.v_left = 0.0
        self.v_right = 0.0
        self.omega_l = 0.0
        self.omega_r = 0.0
        
        # Variables to account for previous step
        self.prev_left = 0.0
        self.prev_right = 0.0
        self.last_t_left = 0.0
        self.last_t_right = 0.0

        # To account for first received message
        self.first_message_left = True
        self.first_message_right = True
        self.initial_ticks_left = 0.0
        self.initial_ticks_right = 0.0

        # Get static parameters
        self._radius = rospy.get_param(f'{self.veh_name}/kinematics_node/radius', 100)
        self._baseline = rospy.get_param('/%s/kinematics_node/baseline', 100)
        self._resolution = 135.0

        # Subscribing to the wheel encoders
        self.sub_encoder_ticks_left = rospy.Subscriber(f'{self.veh_name}/left_wheel_encoder_node/tick',WheelEncoderStamped,callback=self.cb_encoder_data,callback_args='left')
        self.sub_encoder_ticks_right = rospy.Subscriber(f'{self.veh_name}/right_wheel_encoder_node/tick',WheelEncoderStamped,callback=self.cb_encoder_data,callback_args='right')

        # Publishers
        self.pub_robot_pose_tf = rospy.Publisher('~encoder_localization/pose_transform',TransformStamped,queue_size=1)

        self.log("Initialized")

    def cb_encoder_data(self, msg, wheel):
        """ 
        Use encoder information to update linear and angular velocites of each wheel
        """
        # Retrieve encoder data
        ticks = msg.data

        # Check if it's the first received message and store initial encoder value and time
        if wheel == 'left' and self.first_message_left == True:
            self.first_message_left = False
            self.initial_ticks_left = ticks
            self.last_t_left = rospy.get_time()
        if wheel == 'right' and self.first_message_right ==True:
            self.first_message_right = False
            self.initial_ticks_right = ticks
            self.last_t_right = rospy.get_time()

        # Compute linear and angular velocity
        if wheel == 'left':
            rel_ticks = ticks - self.initial_ticks_left
            diff_ticks = rel_ticks - self.prev_left
            dist = 2 * np.pi * self._radius * (diff_ticks / self._resolution)
            dt = rospy.get_time() - self.last_t_left
            # Obtain linear and angular velocity of left wheel
            self.v_left = dist / dt
            self.omega_l = self.v_left / self._radius 
            # Update previous number of ticks
            self.prev_left = rel_ticks

        elif wheel == 'right':
            rel_ticks = ticks - self.initial_ticks_right
            diff_ticks = np.abs(rel_ticks - self.prev_right)
            dist = 2 * np.pi * self._radius * (diff_ticks / self._resolution)
            dt = rospy.get_time() - self.last_t_left
            # Obtain linear and angular velocity of right wheel
            self.v_right = dist / dt
            self.omega_r = self.v_right / self._radius
            # Update previous number of ticks
            self.prev_right = rel_ticks
        
        ####### UPDATE STATE ##########

        # Compute linear and angular velocity of the robot
        self.twist.v = (self.v_left + self.v_right) / 2.0
        self.twist.omega = (self.v_left - self.v_right) / self._baseline
        # Compute current position
        dt = rospy.get_time() - self.last_t
        self.pose.x = self.pose.x + (self.twist.v * dt) * np.cos(self.pose.theta)
        self.pose.y = self.pose.y + (self.twist.v * dt) * np.sin(self.pose.theta)
        self.pose.theta = self.twist.omega * dt

        ###### PUBLISH TRANSFORM MESSAGE ########
        self.current_state.header.stamp = msg.header.stamp
             
        
if __name__ == '__main__':
    node = EncoderLocalizationNode(node_name= 'encoder_localization_node')
    # Keep it spinning to keep the node alive
    rospy.loginfo("encoder_localization_node is up and running...")
    rospy.spin()