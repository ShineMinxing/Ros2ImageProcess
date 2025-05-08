#!/usr/bin/env python3
import os, glob
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray
from ament_index_python.packages import get_package_share_directory
import cv2
import face_recognition
import numpy as np

class FaceRecognitionNode(Node):
    def __init__(self, options=None):
        super().__init__('face_check_node', options=options)

        # 声明并获取参数
        self.image_input_topic = self.declare_parameter(
            'IMAGE_INPUT_TOPIC', '/SMX/GimbalCamera').value
        self.name_output_topic = self.declare_parameter(
            'NAME_OUTPUT_TOPIC', '/SMX/TargetCategory').value
        self.image_output_topic = self.declare_parameter(
            'IMAGE_OUTPUT_TOPIC', '/SMX/TargetImage').value
        self.angle_output_topic = self.declare_parameter(
            'ANGLE_OUTPUT_TOPIC', '/SMX/TargetImageAngle').value
        self.fov_h = float(self.declare_parameter('FOV_H', 125.0).value)
        self.fov_v = float(self.declare_parameter('FOV_V',  69.0).value)
        self.face_lib_dirs = self.declare_parameter(
            'FACE_LIB_DIRS', ['other','local_file']).value

        # 创建订阅 & 发布
        self.image_sub = self.create_subscription(
            Image, self.image_input_topic, self.image_cb, 10)
        self.name_pub  = self.create_publisher(String,             self.name_output_topic,  10)
        self.image_pub = self.create_publisher(Image,              self.image_output_topic, 10)
        self.angle_pub = self.create_publisher(Float64MultiArray, self.angle_output_topic, 10)

        self.bridge = CvBridge()
        self.frame_id = 0

        # 加载人脸库
        pkg_share = get_package_share_directory('face_check')
        self.known_encodings = []
        self.known_names     = []
        for subdir in self.face_lib_dirs:
            full_dir = os.path.join(pkg_share, subdir)
            self.get_logger().info(f'Loading faces from: {full_dir}')
            for img_path in glob.glob(os.path.join(full_dir, '*.*')):
                name = os.path.splitext(os.path.basename(img_path))[0]
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    self.known_encodings.append(encs[0])
                    self.known_names.append(name)
                else:
                    self.get_logger().warn(f'No face found in {img_path}')
        self.get_logger().info(f'Total loaded faces: {len(self.known_names)}')

    def image_cb(self, msg: Image):
        self.frame_id += 1
        if self.frame_id % 3 != 0:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is None:
            self.get_logger().warn('Empty image frame')
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        h, w = frame.shape[:2]
        for (top,right,bottom,left), enc in zip(locs, encs):
            dists = face_recognition.face_distance(self.known_encodings, enc)
            idx = np.argmin(dists) if len(dists) else None
            name = self.known_names[idx] if (idx is not None and dists[idx]<0.55) else 'Unknown'
            cv2.rectangle(frame, (left,top),(right,bottom),(0,255,0),2)
            cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cx = (left+right)/2.0; cy = (top+bottom)/2.0
            rx = (cx - w/2.0)/(w/2.0); ry = -(cy - h/2.0)/(h/2.0)
            ang_x = rx*(self.fov_h/2.0); ang_y = ry*(self.fov_v/2.0)
            # 发布
            self.name_pub.publish(String(data=name))
            a = Float64MultiArray(); a.data=[float(ang_x), float(ang_y), 0.0]
            self.angle_pub.publish(a)
        # 发布标记后图像
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame,'bgr8'))


def main():
    rclpy.init()
    # 加载相对路径的 config.yaml
    options = rclpy.node.NodeOptions()
    options.arguments([ '--ros-args',
                        '--params-file',
                        'src/Ros2ImageProcess/config.yaml' ])
    node = FaceRecognitionNode(options)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()