#!/usr/bin/env python3
import os
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import torch
from torchvision import models, transforms
import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory

class DroneInferenceNode(Node):
    def __init__(self):
        super().__init__('drone_detector_node')

        # 读取参数
        self.image_input_topic = self.declare_parameter(
            'IMAGE_INPUT_TOPIC', '/NoYamlRead/GimbalCamera').value
        self.angle_output_topic = self.declare_parameter(
            'ANGLE_OUTPUT_TOPIC', '/NoYamlRead/TargetImageAngle').value
        self.image_output_topic = self.declare_parameter(
            'IMAGE_OUTPUT_TOPIC', '/NoYamlRead/TargetImage').value
        rel_model_path = self.declare_parameter(
            'MODEL_REL_PATH', 'resource/drone_model_best_0.pth').value
        self.fov_h = float(self.declare_parameter('FOV_H', 125.0).value)
        self.fov_v = float(self.declare_parameter('FOV_V',  69.0).value)

        # 创建订阅和发布
        self.image_sub_ = self.create_subscription(
            Image, self.image_input_topic, self.image_callback, 10)
        self.angle_pub_ = self.create_publisher(
            Float64MultiArray, self.angle_output_topic, 10)
        self.video_pub_ = self.create_publisher(
            Image, self.image_output_topic, 10)

        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # 加载模型
        share_dir = get_package_share_directory('drone_detector')
        model_path = os.path.join(share_dir, rel_model_path)
        self.model = self.load_model(model_path)

        # 图像预处理
        self.data_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def load_model(self, model_path):
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 3)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        self.get_logger().info(f'Model loaded from {model_path}')
        return model

    def image_callback(self, msg: Image):
        # 转 OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = cv_image.shape[:2]

        # 预处理
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(img_rgb)
        inp = self.data_transforms(pil).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            out = self.model(inp)[0].tolist()
        x_n, y_n, t_n = out

        # 还原坐标
        sx, sy = w / 256.0, h / 256.0
        x0, y0 = x_n * sx, y_n * sy
        pt = (int(round(x0)), int(round(y0)))
        cv2.circle(cv_image, pt, 10, (0, 255, 0), 2)

        # 发布标记后图像
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.video_pub_.publish(img_msg)

        self.get_logger().info(f"orig size: h={h}, w={w}")    
        self.get_logger().info(f"model out: x_n={x_n:.4f}, y_n={y_n:.4f}, t_n={t_n:.4f}")
        self.get_logger().info(f"pixel pt: {pt}")

        # 计算角度
        dx = x0 - w / 2.0
        dy = y0 - h / 2.0
        rx = dx / (w / 2.0)
        ry = dy / (h / 2.0)
        ang_x = rx * (self.fov_h / 2.0)
        ang_y = ry * (self.fov_v / 2.0)
        tilt = t_n / 180.0 * math.pi
        tilt_deg = tilt * (180.0 / math.pi)

        # 发布角度
        a = Float64MultiArray()
        a.data = [float(ang_x), float(ang_y), float(tilt_deg)]
        self.angle_pub_.publish(a)


def main(args=None):
    # 通过 init 传入参数文件
    param_file = '/home/unitree/ros2_ws/LeggedRobot/src/Ros2ImageProcess/config.yaml'
    rclpy.init(args=['--ros-args', '--params-file', param_file])

    node = DroneInferenceNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
