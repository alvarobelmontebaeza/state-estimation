#!/usr/bin/env python3
import numpy as np
import os
import math

import rospy
from duckietown.dtros import DTROS, NodeType
from geometry_msgs.msg import TransformStamped, Pose2D, Vector3
import tf, tf2_ros
import tf_conversions

from fused_localization.srv import *

class FusedLocalizationNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(FusedLocalizationNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        # Reduce speed to avoid drifting
        self.default_gain = rospy.get_param('/' + self.veh +'/kinematics_node/gain',default=1.0) 
        rospy.set_param('/' + self.veh +'/kinematics_node/gain', 0.2)

        # State variable for the robot
        self.fused_pose = Pose2D(0.27,0.0,np.pi) # Initial state given arbitrarily

        # State estimates from encoder for current and previous message
        self.encoder_tf = TransformStamped()
        self.prev_encoder_tf = TransformStamped() # To obtain distance traveled

        # Fused transformation to publish/broadcast
        self.fused_pose_transform = TransformStamped()
        self.fused_pose_transform.header.frame_id = 'map'
        self.fused_pose_transform.child_frame_id = 'fused_baselink'
        self.fused_pose_transform.transform.translation = Vector3(self.fused_pose.x,self.fused_pose.y, 0.0)
        q = tf_conversions.transformations.quaternion_from_euler(0,0,self.fused_pose.theta)
        self.fused_pose_transform.transform.rotation.x = q[0]
        self.fused_pose_transform.transform.rotation.y = q[1]
        self.fused_pose_transform.transform.rotation.z = q[2]
        self.fused_pose_transform.transform.rotation.w = q[3]

        # To account for first apriltag detected and first encoder reading
        self.first_apriltag = True
        self.first_encoder = True

        # To account for detected apriltag and choose how to update fused_pose
        self.apriltag_detected = False

        # Subscribers for Apriltag Localization and Encoder Localization
        self.apriltag_sub = rospy.Subscriber('/'+ self.veh + '/at_localization_node/at_baselink_transform', TransformStamped, self.apriltag_cb)
        self.encoder_sub = rospy.Subscriber('/'+ self.veh + '/encoder_localization_node/encoder_baselink_transform', TransformStamped, self.encoder_cb)

        # Publishers and broadcasters
        self.pub_fused_tf = rospy.Publisher('~fused_baselink_transform' , TransformStamped, queue_size=1)
        self.tfBroadcaster = tf.TransformBroadcaster(queue_size=1)

        # Broadcast fused pose tf once
        self.tfBroadcaster.sendTransformMessage(self.fused_pose_transform)

        # Server client to update encoder estimate
        rospy.wait_for_service('/'+ self.veh + '/update_encoder_estimate')
        try:
            self.update_encoder_srv = rospy.ServiceProxy('/'+ self.veh + '/update_encoder_estimate', CalibratePose)
            self.log('SERVICE CLIENT CREATED')
        except rospy.ServiceException as e:
            rospy.logerr('Service call failed: %s'%e)
        
        # Notify complete initialization
        self.log(node_name + ' initialized and running')
    
    def apriltag_cb(self, at_tf):
        # Convert 3D to 2D pose assuming Z = 0 and set fused pose transform
        self.fused_pose_transform.header.stamp = rospy.Time.now()
        self.fused_pose_transform.transform.translation = at_tf.transform.translation
        self.fused_pose_transform.transform.translation.z = 0.0
        q = at_tf.transform.rotation
        angles = tf_conversions.transformations.euler_from_quaternion(np.array([q.x,q.y,q.z,q.w]))
        q = tf_conversions.transformations.quaternion_from_euler(0.0,0.0,angles[2])
        self.fused_pose_transform.transform.rotation.x = q[0]
        self.fused_pose_transform.transform.rotation.y = q[1]
        self.fused_pose_transform.transform.rotation.z = q[2]
        self.fused_pose_transform.transform.rotation.w = q[3]

        if self.first_apriltag:
            # Call service and update encoder estimate
            try:
                resp = self.update_encoder_srv(self.fused_pose_transform)
                self.log('SERVICE CALLED')
                self.first_apriltag = False
            except rospy.ServiceException as e:
                rospy.logerr('Service call failed: %s'%e)
        
        # Update fused pose
        self.fused_pose.x = self.fused_pose_transform.transform.translation.x
        self.fused_pose.y = self.fused_pose_transform.transform.translation.y
        self.fused_pose.theta = angles[2]
        
        # Publish estimate and broadcast every time an apriltag is detected
        self.pub_fused_tf.publish(self.fused_pose_transform)
        self.tfBroadcaster.sendTransformMessage(self.fused_pose_transform)

        # Mark that state has been updated with apriltag
        self.apriltag_detected = True


    def encoder_cb(self, enc_tf):
        if self.first_encoder:
            # Get first encoder estimate
            self.first_encoder = False
            self.prev_encoder_tf = enc_tf

        # Update encoder estimation
        self.prev_encoder_tf = self.encoder_tf
        self.encoder_tf = enc_tf

        # Check if state needs to be updated using encoder
        if self.apriltag_detected == True:
            self.apriltag_detected = False
            return
        else:
            # Obtain displacement measured by encoders
            diff_x = self.encoder_tf.transform.translation.x - self.prev_encoder_tf.transform.translation.x
            diff_y = self.encoder_tf.transform.translation.y - self.prev_encoder_tf.transform.translation.y
            curr_q = self.encoder_tf.transform.rotation
            prev_q = self.prev_encoder_tf.transform.rotation
            curr_angles = tf_conversions.transformations.euler_from_quaternion(np.array([curr_q.x,curr_q.y,curr_q.z,curr_q.w]))
            prev_angles = tf_conversions.transformations.euler_from_quaternion(np.array([prev_q.x,prev_q.y,prev_q.z,prev_q.w]))
            diff_theta = curr_angles[2] - prev_angles[2]
            # Update fused pose
            self.fused_pose.x += diff_x
            self.fused_pose.y += diff_y
            self.fused_pose.theta += diff_theta
            # Update fused pose transform
            self.fused_pose_transform.header.stamp = rospy.Time.now()
            self.fused_pose_transform.transform.translation.x = self.fused_pose.x
            self.fused_pose_transform.transform.translation.y = self.fused_pose.y
            self.fused_pose_transform.transform.translation.z = 0.0
            q = tf_conversions.transformations.quaternion_from_euler(0.0,0.0,self.fused_pose.theta)
            self.fused_pose_transform.transform.rotation.x = q[0]
            self.fused_pose_transform.transform.rotation.y = q[1]
            self.fused_pose_transform.transform.rotation.z = q[2]
            self.fused_pose_transform.transform.rotation.w = q[3]

            # Publish estimate and broadcast
            self.pub_fused_tf.publish(self.fused_pose_transform)
            self.tfBroadcaster.sendTransformMessage(self.fused_pose_transform)

    def onShutdown(self):
        # Restore gain to default value
        rospy.set_param('/' + self.veh +'/kinematics_node/gain', self.default_gain)
        super(FusedLocalizationNode, self).onShutdown()


if __name__ == '__main__':
    # Initialize the node
    fused_node = FusedLocalizationNode(node_name='fused_localization_node')
    # Keep it spinning to keep the node alive
    rospy.spin()