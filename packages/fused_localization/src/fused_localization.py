#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
import yaml
from dt_apriltags import Detector

import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, Pose2D
import tf, tf2_ros
import tf_conversions

class FusedLocalizationNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(FusedLocalizationNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        # State variable for the robot
        self.pose = Pose2D(0.27,0.0,np.pi) # Initial state given arbitrarily

        # Static transforms
        # Map -> Baselink. To be computed 
        self.T_map_baselink = TransformStamped()
        self.T_map_baselink.header.frame_id = 'map'
        self.T_map_baselink.child_frame_id = 'at_baselink'
        # Map -> Apriltag. STATIC
        self._T_map_apriltag = TransformStamped() 
        self._T_map_apriltag.header.frame_id = 'map'
        self._T_map_apriltag.header.stamp = rospy.Time.now()
        self._T_map_apriltag.child_frame_id = 'apriltag'
        self._T_map_apriltag.transform.translation.x = 0.0
        self._T_map_apriltag.transform.translation.y = 0.0
        self._T_map_apriltag.transform.translation.z = 0.09 # Height of 9 cm
        q = tf_conversions.transformations.quaternion_from_euler(0.0,0.0,0.0)
        self._T_map_apriltag.transform.rotation.x = q[0]
        self._T_map_apriltag.transform.rotation.y = q[1]
        self._T_map_apriltag.transform.rotation.z = q[2]
        self._T_map_apriltag.transform.rotation.w = q[3]
        # Apriltag -> Camera. Computed using apriltag detection
        self.T_apriltag_camera = TransformStamped()
        self.T_apriltag_camera.header.frame_id = 'apriltag'
        self.T_apriltag_camera.child_frame_id = 'camera'
        # Camera -> Baselink. STATIC
        self._T_camera_baselink = TransformStamped()
        self._T_camera_baselink.header.frame_id = 'camera'
        self._T_camera_baselink.header.stamp = rospy.Time.now()
        self._T_camera_baselink.child_frame_id = 'at_baselink'
        self._T_camera_baselink.transform.translation.x = -0.0582
        self._T_camera_baselink.transform.translation.y = 0.0
        self._T_camera_baselink.transform.translation.z = -0.110
        q = tf_conversions.transformations.quaternion_from_euler(0.0,np.deg2rad(-15),0.0)
        self._T_camera_baselink.transform.rotation.x = q[0]
        self._T_camera_baselink.transform.rotation.y = q[1]
        self._T_camera_baselink.transform.rotation.z = q[2]
        self._T_camera_baselink.transform.rotation.w = q[3]

        # Transformation Matrices to ease computation
        R_MA = tf_conversions.transformations.euler_matrix(0.0,0.0,0.0,'rxyz')
        t_MA = tf_conversions.transformations.translation_matrix(np.array([0.0,0.0,0.09]))
        self.T_MA = tf_conversions.transformations.concatenate_matrices(t_MA,R_MA)

        R_CB = tf_conversions.transformations.euler_matrix(0.0,np.deg2rad(-15.0),0.0,'rxyz')
        t_CB = tf_conversions.transformations.translation_matrix(np.array([-0.0582,0.0,-0.1072]))
        self.T_CB = tf_conversions.transformations.concatenate_matrices(t_CB,R_CB)
        
        # Define rotation matrices to follow axis convention in rviz when using apriltag detection
        self.T_ApA = tf_conversions.transformations.euler_matrix(np.pi/2.0,0.0, -np.pi/2.0,'rxyz')
        self.T_CCp = tf_conversions.transformations.euler_matrix(-np.pi/2.0,0.0,-np.pi/2.0,'rzyx')

        # Load calibration files
        self.calib_data = self.readYamlFile('/data/config/calibrations/camera_intrinsic/' + self.veh + '.yaml')
        self.log('Loaded intrinsics calibration file')
        self.extrinsics = self.readYamlFile('/data/config/calibrations/camera_extrinsic/' + self.veh + '.yaml')
        self.log('Loaded extrinsics calibration file') 

        # Retrieve intrinsic info
        cam_info = self.setCamInfo(self.calib_data)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(cam_info)
        # Initiate maps for rectification
        self._init_rectify_maps()

        # Create AprilTag detector object
        self.at_detector = Detector()

        # Create cv_bridge
        self.bridge = CvBridge()

        # Define subscriber to recieve images
        self.image_sub = rospy.Subscriber('/' + self.veh+ '/camera_node/image/compressed', CompressedImage, self.callback)
        # Publishers and broadcasters
        self.pub_robot_pose_tf = rospy.Publisher('~pose_transform' , TransformStamped, queue_size=1)
        self.static_tf_br = tf2_ros.StaticTransformBroadcaster()
        self.tfBroadcaster = tf.TransformBroadcaster(queue_size=1)


        self.log(node_name + ' initialized and running')

    def callback(self, ros_image):
        '''
            MAIN TASK
            Convert the image from the camera_node to a cv2 image. Then, detects 
            the apriltags position and retrieves the Rotation and translation to it from the
            camera frame, and next computes T_map_baselink by concatenating transformations.
            The resultant transformation is published, as well as all intermedium are broadcasted
        '''
        # Convert to cv2 image
        image = self.readImage(ros_image)
        # Rectify image
        image = self.processImage(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract camera parameters
        K = np.array(self.cam_model.K).reshape((3,3))
        cam_params = (K[0,0], K[1,1], K[0,2], K[1,2])

        # Detect apriltags
        detections = self.at_detector.detect(gray_image, estimate_tag_pose=True, camera_params=cam_params, tag_size=0.065)        

        # If an apriltag is detected, update transformations
        for tag in detections:
            # Retrieve rotation and translation from apriltag to camera and homogeneize
            R_CA = tf_conversions.transformations.identity_matrix()
            R_CA[:3,:3] = tag.pose_R
            p = tag.pose_t.T[0]
            t_CA = tf_conversions.transformations.translation_matrix(np.array([p[0],p[1],p[2],1.0]))
            # Define homogeneous transformation matrix
            T_CpAp = tf_conversions.transformations.concatenate_matrices(t_CA, R_CA)

            # Compute apriltag->camera transform
            self.T_CA = tf_conversions.transformations.concatenate_matrices(self.T_CCp, T_CpAp, self.T_ApA)
            self.T_AC = tf_conversions.transformations.inverse_matrix(self.T_CA)

            # Compute map->baselink transform
            self.T_MB = tf_conversions.transformations.concatenate_matrices(
                self.T_MA,self.T_AC,self.T_CB
            )

            # Obtain TransformStamped messages
            self.T_apriltag_camera.header.stamp = ros_image.header.stamp
            t = tf_conversions.transformations.translation_from_matrix(self.T_AC)
            self.T_apriltag_camera.transform.translation.x = t[0]
            self.T_apriltag_camera.transform.translation.y = t[1]
            self.T_apriltag_camera.transform.translation.z = t[2]
            q = tf_conversions.transformations.quaternion_from_matrix(self.T_AC)
            self.T_apriltag_camera.transform.rotation.x = q[0]
            self.T_apriltag_camera.transform.rotation.y = q[1]
            self.T_apriltag_camera.transform.rotation.z = q[2]
            self.T_apriltag_camera.transform.rotation.w = q[3]
            
            self.T_map_baselink.header.stamp = ros_image.header.stamp
            t = tf_conversions.transformations.translation_from_matrix(self.T_MB)
            self.T_map_baselink.transform.translation.x = t[0]
            self.T_map_baselink.transform.translation.y = t[1]
            self.T_map_baselink.transform.translation.z = t[2]
            q = tf_conversions.transformations.quaternion_from_matrix(self.T_MB)           
            self.T_map_baselink.transform.rotation.x = q[0]
            self.T_map_baselink.transform.rotation.y = q[1]
            self.T_map_baselink.transform.rotation.z = q[2]
            self.T_map_baselink.transform.rotation.w = q[3]

            # Publish transform map -> baselink
            self.pub_robot_pose_tf.publish(self.T_map_baselink)
            # Broadcast all transforms
            self.static_tf_br.sendTransform(self._T_map_apriltag)
            self.static_tf_br.sendTransform(self._T_camera_baselink)
            self.tfBroadcaster.sendTransformMessage(self.T_apriltag_camera)



    def setCamInfo(self, calib_data):
        '''
            Introduces the camera information from the dictionary
            obtained reading the yaml file to a CameraInfo object
        ''' 
        cam_info = CameraInfo()
        cam_info.width = calib_data['image_width']
        cam_info.height = calib_data['image_height']
        cam_info.K = calib_data['camera_matrix']['data']
        cam_info.D = calib_data['distortion_coefficients']['data']
        cam_info.R = calib_data['rectification_matrix']['data']
        cam_info.P = calib_data['projection_matrix']['data']
        cam_info.distortion_model = calib_data['distortion_model']
        return cam_info  


    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []
    
    def _init_rectify_maps(self):
        # Get new optimal camera matrix
        W = self.cam_model.width
        H = self.cam_model.height
        rect_camera_K, _ = cv2.getOptimalNewCameraMatrix(
            self.cam_model.K,
            self.cam_model.D,
            (W,H),
            1.0
        )
        # Initialize rectification maps
        mapx = np.ndarray(shape=(H,W,1), dtype='float32')
        mapy = np.ndarray(shape=(H,W,1), dtype='float32')
        mapx, mapy = cv2.initUndistortRectifyMap(self.cam_model.K, self.cam_model.D,np.eye(3),rect_camera_K,(W,H),cv2.CV_32FC1,mapx,mapy)
        self.cam_model.K = rect_camera_K
        self.mapx = mapx
        self.mapy = mapy       
    
    def processImage(self, raw_image, interpolation=cv2.INTER_LINEAR):
        '''
        Undistort a provided image using the calibrated camera info
        Implementation based on: https://github.com/duckietown/dt-core/blob/952ebf205623a2a8317fcb9b922717bd4ea43c98/packages/image_processing/include/image_processing/rectification.py
        Args:
            raw_image: A CV image to be rectified
            interpolation: Type of interpolation. For more accuracy, use another cv2 provided constant
        Return:
            Undistorted image
        '''
        image_rectified = np.empty_like(raw_image)
        processed_image = cv2.remap(raw_image,self.mapx,self.mapy,interpolation,image_rectified)

        return processed_image


    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    def onShutdown(self):
        super(FusedLocalizationNode, self).onShutdown()


if __name__ == '__main__':
    # Initialize the node
    fused_node = FusedLocalizationNode(node_name='fused_localization_node')
    # Keep it spinning to keep the node alive
    rospy.spin()