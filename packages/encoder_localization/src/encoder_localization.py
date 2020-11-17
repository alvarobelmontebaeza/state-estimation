#!/usr/bin/env python3
import numpy as np
import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import WheelEncoderStamped
from geometry_msgs.msg import TransformStamped, Pose2D
import tf
import tf_conversions

class EncoderLocalizationNode(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

        # Initialize the DTROS parent class
        super(EncoderLocalizationNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh_name = os.environ['VEHICLE_NAME']

        # State variable for the robot
        self.pose = Pose2D(0.27,0.0,np.pi) # Initial state given arbitrarily

        # Transform that defines robot state
        self.current_state = TransformStamped()
        self.current_state.header.frame_id = 'map'
        self.current_state.child_frame_id = 'encoder_baselink'
        self.current_state.transform.translation.z = 0.0 # We operate in a 2D world

        # Distance traveled by each wheel computed using encoder data
        self.d_left = 0.0
        self.d_right = 0.0

        # Variables to account for previous step
        self.prev_left = 0.0
        self.prev_right = 0.0

        # To account for first received message
        self.first_message_left = True
        self.first_message_right = True
        self.initial_ticks_left = 0.0
        self.initial_ticks_right = 0.0

        # Get static parameters
        self._radius = rospy.get_param('/' + self.veh_name + '/kinematics_node/radius', 0.031)
        self._baseline = rospy.get_param('/' + self.veh_name + '/kinematics_node/baseline', 0.1)
        self._resolution = 135.0

        # Subscribing to the wheel encoders
        self.sub_encoder_ticks_left = rospy.Subscriber('/' + self.veh_name + '/left_wheel_encoder_node/tick',WheelEncoderStamped,callback=self.cb_encoder_data,callback_args='left')
        self.sub_encoder_ticks_right = rospy.Subscriber('/' + self.veh_name + '/right_wheel_encoder_node/tick',WheelEncoderStamped,callback=self.cb_encoder_data,callback_args='right')

        # Publishers
        self.pub_robot_pose_tf = rospy.Publisher('~encoder_baselink_transform',TransformStamped,queue_size=1)
        self.tfBroadcaster = tf.TransformBroadcaster(queue_size=1)
        # Define timer to publish messages at a 30 Hz frequency
        self.pub_timer = rospy.Timer(rospy.Duration(1.0/30.0), self.publish_transform)

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
            return
        if wheel == 'right' and self.first_message_right ==True:
            self.first_message_right = False
            self.initial_ticks_right = ticks
            return

        # Compute distance traveled
        if wheel == 'left':
            rel_ticks = ticks - self.initial_ticks_left
            diff_ticks = rel_ticks - self.prev_left
            dist = 2 * np.pi * self._radius * (diff_ticks / self._resolution)
            # Obtain distance traveled by left wheel
            self.d_left = dist
            # Update previous number of ticks
            self.prev_left = rel_ticks

        elif wheel == 'right':
            rel_ticks = ticks - self.initial_ticks_right
            diff_ticks = rel_ticks - self.prev_right
            dist = 2 * np.pi * self._radius * (diff_ticks / self._resolution)
            # Obtain distance traveled by right wheel
            self.d_right = dist
            # Update previous number of ticks
            self.prev_right = rel_ticks
        
        ####### UPDATE STATE ##########
        d = (self.d_right + self.d_left) / 2.0
        delta_theta = (self.d_right - self.d_left) / self._baseline
        # Compute current position estimate
        self.pose.x = self.pose.x + d * np.cos(self.pose.theta)
        self.pose.y = self.pose.y + d * np.sin(self.pose.theta)
        self.pose.theta = self.pose.theta + delta_theta

    
    def publish_transform(self, timer):
        '''
        Callback method for ROS timer to publish messages at a fixed rate
        '''
        ###### PUBLISH TRANSFORM MESSAGE ########
        self.current_state.header.stamp = rospy.Time.now()
        # Update transform message
        # Translation
        self.current_state.transform.translation.x = self.pose.x
        self.current_state.transform.translation.y = self.pose.y
        # Orientation
        # Transform yaw to quaternion
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, self.pose.theta)
        self.current_state.transform.rotation.x = q[0]
        self.current_state.transform.rotation.y = q[1]
        self.current_state.transform.rotation.z = q[2]
        self.current_state.transform.rotation.w = q[3]

        # Publish and broadcast the transform
        self.pub_robot_pose_tf.publish(self.current_state)
        self.tfBroadcaster.sendTransformMessage(self.current_state)

    def onShutdown(self):
        self.pub_timer.shutdown()
        super(EncoderLocalizationNode, self).onShutdown()
             
        
if __name__ == '__main__':
    node = EncoderLocalizationNode(node_name= 'encoder_localization_node')
    # Keep it spinning to keep the node alive
    rospy.loginfo("encoder_localization_node is up and running...")
    rospy.spin()
