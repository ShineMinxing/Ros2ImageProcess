# Ros2ImageProcess 🖼️

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Ros2ImageProcess** 是基于 **ROS 2 Humble / Ubuntu 22.04** 的 **视觉处理与摄像头驱动仓库**，负责

* 将 *IP Camera / USB Camera / 本地视频* 流快速封装为 ROS `sensor_msgs/Image`
* 提供 *光点/人脸/无人机* 等多目标检测，并输出角度 (`SMX/TargetImageAngle`)
* 支持离线 YOLOv5/v12、OpenCV face rec 与亮度阈值算法，无须外网即可运行
* 支持离线 YOLOv5/v12、OpenCV face rec 与亮度阈值算法，无须外网即可运行

---

## ✨ 功能特性

| 类别                   | 说明                                                                      |
| -------------------- | ----------------------------------------------------------------------- |
| **多源输入**             | `ip_camera`、`usb_camera`、`video_play` 三种节点；支持 RTSP、V4L2、MP4 文件          |
| **实时检测**             | `spot_detector` 光点 / `face_check` 人脸 / `drone_detector` 无人机（YOLO 自训练权重） |
| **角度输出**             | 根据相机视场角 (`FOV_H / FOV_V`) 计算目标偏角，发布 `SMX/TargetImageAngle`              |
| **TF 广播**             | yolo_obb 自动广播云台在 map 下的位姿（gimbal_location + gimbal_orientation）        |
| **Raw & Compressed** | 同时发布 `*_Raw` 与 `*_Compressed` 话题，兼容低带宽传输                                |
| **扩展友好**             | 任意新检测模型仅需订阅图像并发布角度即可，与下游控制逻辑解耦                                          |

---

## 🏗️ 生态仓库详细信息参见

[https://github.com/ShineMinxing/Ros2Go2Estimator](https://github.com/ShineMinxing/Ros2Go2Estimator)

---

## 📂 本仓库结构

```
Ros2ImageProcess/
├── ip_camera/          # RTSP → Image             
├── usb_camera/         # V4L2 → Image            
├── video_play/         # MP4 → Image             
├── spot_detector/      # 红/绿光点检测
├── face_check/         # 人脸识别 + 角度           
├── yolo_obb/           # YOLOv11-OBB 推理 + 观测数组 + TF
├── drone_detector/     # 自训练无人机权重示例        
├── config.yaml         # 全局参数
└── Readme.md           # ← 你正在看
```

---

## ⚙️ 关键参数 `config.yaml`

| 节点                        | 关键参数                                | 默认值                        | 说明                     |
| ------------------------- | ----------------------------------- | -------------------------- | ---------------------- |
| **ip\_camera\_node**      | `IP_GSTREAMER`                      | 详见 yaml                   | 自定义 GStreamer Pipeline |
| **usb\_camera\_node**     | `device_id / publish_fps`           | `4 / 30`                   | USB 摄像头编号 / 帧率       |
| **video\_play\_node**     | `VIDEO_FILE_PATH / PUBLISH_FPS`     | `~/Video.mp4 / 30`         | 本地视频路径 / 发布帧率      |
| **spot\_detector\_node**  | `IMAGE_INPUT_TOPIC / FOV_H / FOV_V` | `/SMX/Camera_Raw / 125/69` | 发布光点角度                |
| **face\_check\_node**     | `FACE_LIB_DIRS`                     | `other & local_file`       | 目录内图片即人脸库           |
| **yolo\_obb\_node**       | `model_path`                        | `best.pt`                  | YOLO 权重                |

---

## 🛠️ 安装与编译

> 已在 NV RTX‑40 系列台式机 & Jetson Orin 测试通过，如需 CUDA 推理请提前装好 TensorRT / CuDNN。

```bash
# 1. 依赖
sudo apt install -y ros-humble-cv-bridge ros-humble-image-transport \
                ros-humble-vision-opencv python3-colcon-common-extensions \
                python3-numpy python3-pip build-essential cmake python3-dev \
                libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev libjpeg-dev \
                libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev

# 可选：dlib + face_recognition
git clone https://github.com/davisking/dlib.git && cd dlib
mkdir build && cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
make -j$(nproc) && sudo make install
pip3 install --user dlib==19.24.4 face_recognition==1.3.0 opencv-python

# 2. clone & build
cd ~/ros2_ws/LeggedRobot/src
git clone --recursive https://github.com/ShineMinxing/Ros2ImageProcess.git
cd .. && colcon build --packages-select ip_camera usb_camera video_play spot_detector face_check yolo_ros drone_detector
source install/setup.bash

# 3. 运行示例
ros2 run ip_camera ip_camera_node   # 或 usb_camera_node / video_play_node
ros2 run spot_detector spot_detector_node
ros2 run yolo_obb yolo_obb_node
```

---

## 📑 主要节点接口速查

```text
/ip_camera_node
  • 输出  /SMX/Camera_Raw   sensor_msgs/Image (BGR)
  • 输出  /SMX/Camera_Compressed sensor_msgs/CompressedImage

/spot_detector_node
  • 订阅  /SMX/Camera_Raw  Image
  • 发布  /SMX/TargetImage Image (标记)
  • 发布  /SMX/TargetImageAngle std_msgs/Float64MultiArray [yaw, pitch]

/face_check_node   (同上，外加 /SMX/TargetCategory)

/yolo_obb_node
  • 订阅  /camera/image_raw  Image
  • 发布  /yolo/annotated   Image (叠加 OBB)
  • 发布  /SMX/YOLO_Obs     Float64MultiArray (N×6观测: 方位角,俯仰角,距离,roll,pitch,置信度)
  • 广播  TF: map -> 节点名 (使用 gimbal_location & gimbal_orientation)
```

---

## 📄 深入阅读

* 训练的yolo模型：[https://pan.quark.cn/s/c31e3ce92149](https://pan.quark.cn/s/c31e3ce92149)
* 技术原理笔记：[https://www.notion.so/Ros2Go2-1e3a3ea29e778044a4c9c35df4c27b22](https://www.notion.so/Ros2Go2-1e3a3ea29e778044a4c9c35df4c27b22)
* ROS1 版本参考：[https://github.com/ShineMinxing/FusionEstimation](https://github.com/ShineMinxing/FusionEstimation)

---

## 📨 联系我们

| 邮箱                                          | 单位           |
| ------------------------------------------- | ------------ |
| [401435318@qq.com](mailto:401435318@qq.com) | 中国科学院光电技术研究所 |

> 📌 **本仓库仍在持续开发中** — 欢迎 Issue / PR 交流、贡献！
