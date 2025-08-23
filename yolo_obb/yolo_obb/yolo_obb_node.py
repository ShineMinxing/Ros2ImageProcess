import os
from pathlib import Path
import time
import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from cv_bridge import CvBridge
from ultralytics import YOLO


# 你希望的参数文件固定路径
DEFAULT_CFG_ABS = "/home/unitree/ros2_ws/LeggedRobot/src/Ros2ImageProcess/config.yaml"


def _guess_config_path() -> str:
    """
    优先使用 DEFAULT_CFG_ABS；若不存在，则从本文件往上找包含 src 的工作区根，
    拼出 src/Ros2ImageProcess/config.yaml；都没有则返回 DEFAULT_CFG_ABS（让错误更清晰）。
    """
    p = Path(DEFAULT_CFG_ABS)
    if p.is_file():
        return str(p)

    # 从当前文件往上找工作区根
    cur = Path(__file__).resolve()
    ws_root = None
    for _ in range(8):
        if (cur / "src").is_dir():
            ws_root = cur
            break
        cur = cur.parent

    if ws_root:
        q = ws_root / "src" / "Ros2ImageProcess" / "config.yaml"
        if q.is_file():
            return str(q)

    return str(p)  # 默认返回固定绝对路径（即使不存在，也便于日志提示）


class YoloObbNode(Node):
    def __init__(self, **kwargs):
        # 让 rcl 根据 params-file 自动覆盖未声明的参数（更宽松）
        kwargs.setdefault("automatically_declare_parameters_from_overrides", True)
        super().__init__('yolo_obb_node', **kwargs)

        # ---------------- 参数默认值（可被 params-file 覆盖） ----------------
        defaults = {
            'model_path': 'yolov11-obb.pt',
            'image_topic': '/camera/image_raw',
            'imgsz': 640,
            'conf': 0.25,
            'device': 'cuda:0',
            'half': True,
            'draw': True,
            'output_image_topic': '/yolo/annotated',
            'output_dets_topic': '/yolo/obb_dets',
            'image_qos': 'reliable',  # reliable | best_effort | sensor
        }
        for k, v in defaults.items():
            if not self.has_parameter(k):  # 被 params-file 覆盖时就不再声明
                self.declare_parameter(k, v)

        # ---------------- 取参（最终值） ----------------
        gp = self.get_parameter
        self.model_path = gp('model_path').value
        self.image_topic = gp('image_topic').value
        self.imgsz = int(gp('imgsz').value)
        self.conf = float(gp('conf').value)
        self.device = gp('device').value
        self.half = bool(gp('half').value)
        self.draw = bool(gp('draw').value)
        self.output_image_topic = gp('output_image_topic').value
        self.output_dets_topic = gp('output_dets_topic').value
        image_qos_mode = str(gp('image_qos').value).strip().lower()

        # ---------------- 设备 ----------------
        if self.device.startswith('cuda') and torch.cuda.is_available():
            try:
                self.get_logger().info(f'Using GPU: {torch.cuda.get_device_name(0)}')
            except Exception:
                self.get_logger().info('Using GPU')
            torch.backends.cudnn.benchmark = True
        else:
            self.get_logger().warn('CUDA 不可用，切换 CPU')
            self.device, self.half = 'cpu', False

        # ---------------- 模型 ----------------
        self.model = YOLO(self.model_path).to(self.device)
        try:
            self.model.fuse()
        except Exception:
            pass

        self.bridge = CvBridge()

        # ---------------- 订阅 QoS ----------------
        if image_qos_mode in ('best_effort', 'besteffort'):
            image_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )
        elif image_qos_mode in ('sensor', 'sensor_data'):
            image_qos = qos_profile_sensor_data
        else:
            image_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
        self.get_logger().info(f'Image QoS mode: {image_qos_mode}')

        self.sub = self.create_subscription(Image, self.image_topic, self.cb_image, image_qos)

        # ---------------- 发布者 ----------------
        self.pub_dets = self.create_publisher(Float64MultiArray, self.output_dets_topic, 10)
        self.pub_img  = self.create_publisher(Image,               self.output_image_topic, 10)

        # ---------------- 统计 ----------------
        self.get_logger().info(f'Subscribing: {self.image_topic}')
        self.get_logger().info(f'Annotated:  {self.output_image_topic}')
        self.get_logger().info(f'Detections: {self.output_dets_topic}')

    @torch.inference_mode()
    def cb_image(self, msg: Image):

        # 1) ROS->CV，并复制成独立缓冲区，避免内存视图副作用
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').copy()

        det_list = []
        try:
            # 2) 推理（task='obb'）
            results = self.model.predict(
                source=cv_img, imgsz=self.imgsz, conf=self.conf,
                device=self.device, verbose=False, task='obb'
            )
            if results:
                res = results[0]
                obb = getattr(res, 'obb', None)
                names = getattr(res, 'names', None)

                if obb is not None:
                    xywhr = getattr(obb, 'xywhr', None)
                    if xywhr is not None:
                        xywhr = xywhr.detach().cpu().numpy()
                        cls_np  = (getattr(obb, 'cls', None).detach().cpu().numpy().astype(np.int32)
                                   if getattr(obb, 'cls', None) is not None else np.zeros((xywhr.shape[0],), np.int32))
                        conf_np = (getattr(obb, 'conf', None).detach().cpu().numpy().astype(np.float32)
                                   if getattr(obb, 'conf', None) is not None else np.ones((xywhr.shape[0],), np.float32))
                        for (cx, cy, w, h, r), c, s in zip(xywhr, cls_np, conf_np):
                            det_list.append([float(cx), float(cy), float(w), float(h), float(r), int(c), float(s)])
                            if self.draw:
                                rect = ((float(cx), float(cy)), (float(w), float(h)), float(np.degrees(r)))
                                box = cv2.boxPoints(rect).astype(np.int32)
                                cv2.polylines(cv_img, [box], True, (0, 255, 0), 2)
                                label = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
                                cv2.putText(cv_img, f'{label}:{s:.2f}', (int(cx), int(cy)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
                    else:
                        polys = getattr(obb, 'xyxyxyxy', None)
                        if polys is not None:
                            p = polys.detach().cpu().numpy()
                            cls_np  = (getattr(obb, 'cls', None).detach().cpu().numpy().astype(np.int32)
                                       if getattr(obb, 'cls', None) is not None else np.zeros((p.shape[0],), np.int32))
                            conf_np = (getattr(obb, 'conf', None).detach().cpu().numpy().astype(np.float32)
                                       if getattr(obb, 'conf', None) is not None else np.ones((p.shape[0],), np.float32))
                            for poly, c, s in zip(p, cls_np, conf_np):
                                pts = poly.reshape(-1, 2).astype(np.float32)
                                (cx, cy), (w, h), deg = cv2.minAreaRect(pts)
                                r = np.radians(deg)
                                det_list.append([float(cx), float(cy), float(w), float(h), float(r), int(c), float(s)])
                                if self.draw:
                                    box = cv2.boxPoints(((cx, cy), (w, h), deg)).astype(np.int32)
                                    cv2.polylines(cv_img, [box], True, (0, 255, 0), 2)
                else:
                    self.get_logger().warn('结果中未发现 obb 字段', throttle_duration_sec=5.0)

        except Exception as e:
            self.get_logger().error(f'推理异常：{e}', throttle_duration_sec=5.0)

        # 3) 发布检测数组
        if det_list:
            dets = np.asarray(det_list, dtype=np.float64)
            out = Float64MultiArray()
            out.layout = MultiArrayLayout(dim=[
                MultiArrayDimension(label='detections', size=dets.shape[0], stride=dets.shape[0]*7),
                MultiArrayDimension(label='fields', size=7, stride=7)
            ])
            out.data = dets.ravel().tolist()
            self.pub_dets.publish(out)

        # 4) 始终发布图像（只要 draw=True）
        if self.draw:
            img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            img_msg.header = msg.header
            self.pub_img.publish(img_msg)


def main():
    rclpy.init()

    # 1) 组装 params-file（优先固定绝对路径，不在则智能回退）
    config_file = _guess_config_path()
    cli_args = []
    if Path(config_file).is_file():
        cli_args = ['--ros-args', '--params-file', config_file]
    else:
        # 找不到就让节点用其默认参数，同时打日志
        print(f"[yolo_obb_node] 警告：未找到配置文件：{config_file}，将使用节点内默认参数")

    # 2) 通过 cli_args 传给节点；允许从 params-file 自动声明未显式声明的参数
    node = YoloObbNode(cli_args=cli_args,
                       automatically_declare_parameters_from_overrides=True)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt，准备退出...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
