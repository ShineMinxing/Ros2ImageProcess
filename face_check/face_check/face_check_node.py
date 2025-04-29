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
    def __init__(self):
        super().__init__('face_check_node')

        # 订阅摄像头
        self.image_sub = self.create_subscription(
            Image, '/SMX/GimbalCamera', self.image_cb, 10)

        # 发布识别结果
        self.name_pub   = self.create_publisher(String, 'SMX/TargetCategory', 10)
        self.image_pub  = self.create_publisher(Image , 'SMX/TargetImage', 10)
        self.angle_pub  = self.create_publisher(Float64MultiArray,
                                                'SMX/TargetImageAngle', 10)
        
        self.frame_id = 0

        # 相机视场角（度）
        self.declare_parameter('fov_h', 125.0)
        self.declare_parameter('fov_v', 69.0)
        self.fov_h = float(self.get_parameter('fov_h').value)
        self.fov_v = float(self.get_parameter('fov_v').value)

        self.bridge = CvBridge()

        # 载入人脸库
        pkg_share = get_package_share_directory('face_check')
        faces_dirs  = [
            os.path.join(pkg_share, 'other'),
            os.path.join(pkg_share, 'local_file')
        ]
        self.known_encodings = []
        self.known_names     = []

        for faces_dir in faces_dirs:
            self.get_logger().info(f'Loading faces from {faces_dir}')
            for img_path in glob.glob(os.path.join(faces_dir, '*.*')):
                name = os.path.splitext(os.path.basename(img_path))[0]
                image = face_recognition.load_image_file(img_path)
                encs  = face_recognition.face_encodings(image)
                if encs:
                    self.known_encodings.append(encs[0])
                    self.known_names.append(name)
                else:
                    self.get_logger().warn(f'No face found in {img_path}')
        self.get_logger().info(f'Total loaded faces: {len(self.known_names)}')

    # ──────────────────────────────────────────────────────────────
    def image_cb(self, msg: Image):
        self.frame_id += 1
        if self.frame_id % 3 != 0:
            return 
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is None:
            self.get_logger().warn('Empty image frame')
            return

        # 1. 检测 + 编码
        # small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # scale = 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb, model='hog', number_of_times_to_upsample=0)
        face_encs = face_recognition.face_encodings(rgb, face_locs)

        height, width = frame.shape[:2]

        for (top, right, bottom, left), enc in zip(face_locs, face_encs):
            # top, right, bottom, left = [v * scale for v in (top, right, bottom, left)]
            # 2. 比对
            distances = face_recognition.face_distance(self.known_encodings, enc)
            idx = np.argmin(distances) if len(distances) else None
            name = self.known_names[idx] if idx is not None and distances[idx] < 0.55 else 'Unknown'

            # 3. 画绿框 + 名字
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # 4. 角度计算（取中心点）
            cx = (left + right) / 2.0
            cy = (top  + bottom) / 2.0
            ratio_x = (cx - width/2)  / (width/2)   # -1~1
            ratio_y = -(cy - height/2) / (height/2) # -1~1
            angle_x = ratio_x * (self.fov_h / 2.0)
            angle_y = ratio_y * (self.fov_v / 2.0)

            # 5. 发布人名
            self.name_pub.publish(String(data=name))

            # 6. 发布角度
            a_msg = Float64MultiArray()
            a_msg.data = [float(angle_x), float(angle_y), 0.0]
            self.angle_pub.publish(a_msg)

        # 7. 发布标记后图像
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))

# ──────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = FaceRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
