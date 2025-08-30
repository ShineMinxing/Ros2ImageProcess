import sys
from pathlib import Path
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
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker
import math

# 固定优先的参数文件（找不到会回退搜索）
DEFAULT_CFG_ABS = "/home/unitree/ros2_ws/LeggedRobot/src/Ros2ImageProcess/config.yaml"


def _guess_config_path() -> str:
    p = Path(DEFAULT_CFG_ABS)
    if p.is_file():
        return str(p)
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
    return str(p)


class YoloObbNode(Node):
    def __init__(self, **kwargs):
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
            'image_qos': 'reliable',  # reliable | best_effort | sensor

            # 输出话题
            'output_image_topic': '/yolo/annotated',
            'output_obs_topic':   '/SMX/YOLO_Obs',   # 5个观测量 + conf

            # 相机 FOV（度）
            'fov_h_deg': 70.0,
            'fov_v_deg': 43.0,

            # 无人机真实宽度（米）
            'uav_width_m': 0.5,

            # pitch 计算比例系数
            'pitch_ratio_k': 3.0,

            # 云台角话题
            'gimbal_angle_topic': 'SMX/GimbalState',
            'vehicle_angle_topic':'SMX/GimbalState',

            # 位姿（弧度）
            'gimbal_location': [0.0, 0.0, 0.0],
            'gimbal_orientation': [0.0, 0.0, 0.0],  # [roll,pitch,yaw] rad

            # 跟踪类别筛选
            'target_names': [],   # 例如 ["uav", "drone"]

            # ===== RViz 可视化（新增） =====
            'rviz_mesh_resource': '',       # 当 model=mesh 时使用（建议 package://yolo_obb/resource/*.stl）
            'rviz_scale': [1.0, 1.0, 1.0],  # 对 mesh/arrow 的缩放
            'rviz_color': [0.1, 0.6, 1.0, 0.9],  # arrow 的 RGBA
            'rviz_ns': 'yolo_obb',
        }
        for k, v in defaults.items():
            if not self.has_parameter(k):
                self.declare_parameter(k, v)

        # ---------------- 取参（最终值） ----------------
        gp = self.get_parameter
        self.model_path          = gp('model_path').value
        self.image_topic         = gp('image_topic').value
        image_qos_mode           = str(gp('image_qos').value).strip().lower()
        self.imgsz               = int(gp('imgsz').value)
        self.conf                = float(gp('conf').value)
        self.device              = gp('device').value
        self.half                = bool(gp('half').value)
        self.draw                = bool(gp('draw').value)
        self.target_names        = list(gp('target_names').value)
        self.pitch_ratio_k       = float(gp('pitch_ratio_k').value)
        self.uav_width_m         = float(gp('uav_width_m').value)
        self.fov_h_deg           = float(gp('fov_h_deg').value)
        self.fov_v_deg           = float(gp('fov_v_deg').value)
        self.gimbal_angle_topic  = str(gp('gimbal_angle_topic').value)
        self.vehicle_angle_topic = str(gp('vehicle_angle_topic').value)
        self.gimbal_location     = [float(v) for v in gp('gimbal_location').value]
        self.gimbal_orientation  = [float(v) for v in gp('gimbal_orientation').value]
        self.output_image_topic  = gp('output_image_topic').value
        self.output_obs_topic    = gp('output_obs_topic').value
        self.rviz_mesh_resource  = str(gp('rviz_mesh_resource').value)
        self.rviz_scale          = [float(x) for x in gp('rviz_scale').value]
        self.rviz_color          = [float(x) for x in gp('rviz_color').value]
        self.rviz_ns             = str(gp('rviz_ns').value)

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

        # 图像与云台订阅
        self.sub = self.create_subscription(Image, self.image_topic, self.cb_image, image_qos)
        self.tf_broadcaster = TransformBroadcaster(self)

        # RViz Marker 发布者
        self.marker_pub = self.create_publisher(Marker, "visualization_marker", 10)

        self.roll_gimbal = self.gimbal_orientation[0]
        self.pitch_gimbal = self.gimbal_orientation[1]
        self.yaw_gimbal = self.gimbal_orientation[2]
        self.sub_gimbal = self.create_subscription(Float64MultiArray, self.gimbal_angle_topic, self.cb_gimbal, 10)

        self.roll_vehicle = 0.0
        self.pitch_vehicle  = 0.0
        self.yaw_vehicle  = 0.0
        self.sub_vehicle = self.create_subscription(Float64MultiArray, self.vehicle_angle_topic, self.cb_vehicle, 10)

        # ---------------- 发布者 ----------------
        self.pub_img  = self.create_publisher(Image,             self.output_image_topic, 10)
        self.pub_obs  = self.create_publisher(Float64MultiArray, self.output_obs_topic,   10)

        self.get_logger().info(f'Subscribing: {self.image_topic}')
        self.get_logger().info(f'Annotated:  {self.output_image_topic}')
        self.get_logger().info(f'Observations: {self.output_obs_topic}')
        self.get_logger().info(f'FOV(H/V):   {self.fov_h_deg:.2f}° / {self.fov_v_deg:.2f}°  UAV_W: {self.uav_width_m:.3f} m')

    # ---------- 几何辅助 ----------
    @staticmethod
    def _normalize_rect(w: float, h: float, r_rad: float):
        if w < h:
            w, h = h, w
            r_rad = r_rad + np.pi/2
        r_rad = (r_rad + np.pi/2) % np.pi - np.pi/2
        return w, h, r_rad

    @staticmethod
    def _proj_hw_from_box(box: np.ndarray):
        xs = box[:, 0]
        ys = box[:, 1]
        W_proj = float(max(1.0, float(xs.max() - xs.min())))
        H_proj = float(max(1.0, float(ys.max() - ys.min())))
        return W_proj, H_proj

    # ---------- TF + Marker ----------
    def _quat_from_rpy(self, roll, pitch, yaw):
        cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
        cp = math.cos(pitch* 0.5);  sp = math.sin(pitch* 0.5)
        cy = math.cos(yaw  * 0.5);  sy = math.sin(yaw  * 0.5)
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        qw = cr*cp*cy + sr*cp*sy
        return qx, qy, qz, qw

    def publish_gimbal_tf_and_marker(self, stamp):
        # 1) TF
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id  = self.get_name()

        tf_msg.transform.translation.x = float(self.gimbal_location[0])
        tf_msg.transform.translation.y = float(self.gimbal_location[1])
        tf_msg.transform.translation.z = float(self.gimbal_location[2])

        roll  = float(self.gimbal_orientation[0])
        pitch = float(self.gimbal_orientation[1])
        yaw   = float(self.gimbal_orientation[2])
        qx, qy, qz, qw = self._quat_from_rpy(roll, pitch, yaw)

        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf_msg)

        # 2) Marker（可选）
        if not self.marker_pub:
            return

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "map"
        marker.ns = self.rviz_ns
        marker.id = 1
        marker.action = Marker.ADD
        marker.pose.position.x = tf_msg.transform.translation.x
        marker.pose.position.y = tf_msg.transform.translation.y
        marker.pose.position.z = tf_msg.transform.translation.z
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        # 常见 mesh 资源（按你的 common_sensors 包调整）
        MESH_TABLE = {
            'kinect':  'package://common_sensors/meshes/kinect.dae',
            'xtion':   'package://common_sensors/meshes/asus_xtion_pro.dae',
            'hokuyo':  'package://common_sensors/meshes/hokuyo_utm30lx.dae',
        }

        marker.type = Marker.MESH_RESOURCE
        path = self.rviz_mesh_resource
        if path and '://' not in path:
            path = 'file://' + path
        marker.mesh_resource = path
        marker.mesh_use_embedded_materials = True
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.95 
        marker.scale.x = float(self.rviz_scale[0])
        marker.scale.y = float(self.rviz_scale[1])
        marker.scale.z = float(self.rviz_scale[2])

        self.marker_pub.publish(marker)

    # ---------------- 云台角回调 ----------------
    def cb_gimbal(self, msg: Float64MultiArray):
        data = msg.data
        try:
            if len(data) >= 3:
                self.roll_gimbal  = float(data[0]) * np.pi / 180.0
                self.pitch_gimbal = float(data[1]) * np.pi / 180.0
                self.yaw_gimbal   = float(data[2]) * np.pi / 180.0
        except Exception as e:
            self.get_logger().warn(f'gimbal 数据异常: {e}')

    # ---------------- 载具角回调 ----------------
    def cb_vehicle(self, msg: Float64MultiArray):
        data = msg.data
        try:
            if len(data) >= 3:
                self.roll_vehicle  = float(data[0]) * np.pi / 180.0
                self.pitch_vehicle = float(data[1]) * np.pi / 180.0
                self.yaw_vehicle   = float(data[2]) * np.pi / 180.0
        except Exception as e:
            self.get_logger().warn(f'gimbal 数据异常: {e}')

    # ---------------- 图像回调 ----------------
    @torch.inference_mode()
    def cb_image(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').copy()
        W_H = float(msg.width)
        W_E = float(msg.height)

        xi_H = float(self.fov_h_deg) * np.pi / 180.0
        xi_E = float(self.fov_v_deg) * np.pi / 180.0

        obs_list = []  # [theta_A_k, theta_A_E, d, eta_roll, eta_pitch, conf]

        BLUE = (255, 0, 0)
        THK  = 1
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FS   = 0.5
        DY   = 16

        try:
            results = self.model.predict(
                source=cv_img, imgsz=self.imgsz, conf=self.conf,
                device=self.device, verbose=False
            )
            if results:
                res = results[0]
                obb = getattr(res, 'obb', None)
                names = getattr(res, 'names', None)

                def _cls_name(cls_id):
                    if isinstance(names, dict):
                        return names.get(int(cls_id), str(int(cls_id)))
                    elif isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
                        return str(names[int(cls_id)])
                    return str(int(cls_id))

                def _is_target(label_str: str) -> bool:
                    if not self.target_names:
                        return True
                    return label_str.strip().lower() in self.target_names

                # 兼容非 OBB 模型：用 axis-aligned boxes 伪造 OBB
                if (obb is None) and hasattr(res, 'boxes') and (res.boxes is not None) and (len(res.boxes) > 0):
                    boxes = res.boxes
                    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)

                    if hasattr(res, 'orig_shape') and isinstance(res.orig_shape, (tuple, list)) and len(res.orig_shape) >= 2:
                        H0, W0 = float(res.orig_shape[0]), float(res.orig_shape[1])
                    elif hasattr(res, 'orig_img') and res.orig_img is not None:
                        H0, W0 = float(res.orig_img.shape[0]), float(res.orig_img.shape[1])
                    else:
                        H0, W0 = float(cv_img.shape[0]), float(cv_img.shape[1])
                    H1, W1 = float(cv_img.shape[0]), float(cv_img.shape[1])
                    if (W0 > 0 and H0 > 0) and ((abs(W0 - W1) > 1e-3) or (abs(H0 - H1) > 1e-3)):
                        sx, sy = W1 / W0, H1 / H0
                        xyxy[:, [0, 2]] *= sx
                        xyxy[:, [1, 3]] *= sy

                    cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
                    cy = 0.5 * (xyxy[:, 1] + xyxy[:, 3])
                    w  = np.maximum(xyxy[:, 2] - xyxy[:, 0], 1e-6)
                    h  = np.maximum(xyxy[:, 3] - xyxy[:, 1], 1e-6)
                    r  = np.zeros_like(w, dtype=np.float32)
                    xywhr_np = np.stack([cx, cy, w, h, r], axis=1).astype(np.float32)

                    from types import SimpleNamespace
                    import torch as _torch
                    cls_np = (boxes.cls.detach().cpu().numpy().astype(np.int32)
                              if getattr(boxes, 'cls', None) is not None else np.zeros((xywhr_np.shape[0],), np.int32))
                    conf_np = (boxes.conf.detach().cpu().numpy().astype(np.float32)
                               if getattr(boxes, 'conf', None) is not None else np.ones((xywhr_np.shape[0],), np.float32))
                    obb = SimpleNamespace(
                        xywhr=_torch.from_numpy(xywhr_np),
                        cls=_torch.from_numpy(cls_np.astype(np.float32)),
                        conf=_torch.from_numpy(conf_np.astype(np.float32)),
                    )

                if obb is not None:
                    xywhr = getattr(obb, 'xywhr', None)
                    if xywhr is not None:
                        xywhr = xywhr.detach().cpu().numpy()
                        cls_np  = (getattr(obb, 'cls', None).detach().cpu().numpy().astype(np.int32)
                                   if getattr(obb, 'cls', None) is not None else np.zeros((xywhr.shape[0],), np.int32))
                        conf_np = (getattr(obb, 'conf', None).detach().cpu().numpy().astype(np.float32)
                                   if getattr(obb, 'conf', None) is not None else np.ones((xywhr.shape[0],), np.float32))

                        for (cx, cy, w, h, r), c, s in zip(xywhr, cls_np, conf_np):
                            label = _cls_name(c)
                            if not _is_target(label):
                                continue

                            w_, h_, r_ = self._normalize_rect(float(w), float(h), float(r))

                            rect = ((float(cx), float(cy)), (w_, h_), float(np.degrees(r_)))
                            box  = cv2.boxPoints(rect).astype(np.int32)
                            W_proj, H_proj = self._proj_hw_from_box(box)

                            # --- 5个观测量 ---
                            self.gimbal_orientation[0] = self.roll_gimbal + self.roll_vehicle
                            self.gimbal_orientation[1] = self.pitch_gimbal + self.pitch_vehicle
                            self.gimbal_orientation[2] = self.yaw_gimbal + self.yaw_vehicle
                            theta_A_k = self.gimbal_orientation[2] + ((2.0*cx*xi_H - W_H*xi_H) / (2.0*W_H))
                            theta_A_E = self.gimbal_orientation[1] - ((2.0*cy*xi_E - W_E*xi_E) / (2.0*W_E))

                            angle_pix = np.clip(W_proj * xi_H / W_H, 1e-6, np.pi/2 - 1e-6)
                            d_k       = float(self.uav_width_m / np.tan(angle_pix))

                            eta_roll  = float(r_)
                            ratio     = float(np.clip((H_proj - self.pitch_ratio_k * W_proj) / ((1 - self.pitch_ratio_k) * W_proj), -1.0, 1.0))
                            eta_pitch = theta_A_k + float(np.arcsin(ratio))
                            obs_list.append([theta_A_k, theta_A_E, d_k, eta_roll, eta_pitch, float(s)])

                            if self.draw:
                                cv2.polylines(cv_img, [box], True, BLUE, THK)
                                x0, y0 = int(cx), int(cy)
                                lines = [
                                    f"{label} {s:.2f}",
                                    f"thA_k:{np.degrees(theta_A_k):.3f}deg",
                                    f"thA_E:{np.degrees(theta_A_E):.3f}deg",
                                    f"d:{d_k:.2f}m",
                                    f"roll:{np.degrees(eta_roll):.3f}deg",
                                    f"pitch:{np.degrees(eta_pitch):.3f}deg",
                                ]
                                for i, txt in enumerate(lines):
                                    cv2.putText(cv_img, txt, (x0, y0 + i*DY), FONT, FS, BLUE, THK, cv2.LINE_AA)
                else:
                    self.get_logger().warn('结果中未发现 obb 字段')

        except Exception as e:
            self.get_logger().error(f'推理异常：{e}')

        # 3) 发布观测数组
        if obs_list:
            obs = np.asarray(obs_list, dtype=np.float64)
            msg_obs = Float64MultiArray()
            msg_obs.layout = MultiArrayLayout(dim=[
                MultiArrayDimension(
                    label=f"location:{self.gimbal_location[0]},{self.gimbal_location[1]},{self.gimbal_location[2]}",
                    size=obs.shape[0],
                    stride=obs.shape[0]*6
                ),
            ])
            msg_obs.data = obs.ravel().tolist()
            self.pub_obs.publish(msg_obs)

        # 4) 发布叠加图像
        if self.draw:
            img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            img_msg.header = msg.header
            self.pub_img.publish(img_msg)

        # 5) TF + RViz 模型
        self.publish_gimbal_tf_and_marker(msg.header.stamp)


def main():
    rclpy.init(args=sys.argv)

    inject_cli = []
    if '--params-file' not in sys.argv:
        default_cfg = _guess_config_path()
        if Path(default_cfg).is_file():
            inject_cli = ['--ros-args', '--params-file', default_cfg]
        else:
            print(f"[yolo_obb_node] 警告：未找到默认配置文件：{default_cfg}，使用节点内默认参数")

    node = YoloObbNode(
        cli_args=inject_cli,
        automatically_declare_parameters_from_overrides=True
    )
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
