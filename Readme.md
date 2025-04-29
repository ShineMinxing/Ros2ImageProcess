## ⚙️ 安装指南

spot_detector使用狗头相机提取绿色光电，输出光电偏角，交给g1_control进行控制
face_check使用吊舱相机识别人脸，输出光电偏角，交给g1_control进行控制

- Use Ubuntu 22.04, ROS2 Humble
```bash
sudo apt install -y ros-humble-cv-bridge ros-humble-image-transport \
                ros-humble-vision-opencv python3-colcon-common-extensions \
                python3-numpy python3-pip build-essential cmake  python3-dev \
                libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev libjpeg-dev \
                libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev

git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build && cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
make -j4      # 根据你 Jetson 的核心数调整并行度
sudo make install
cd ../

pip3 install --user dlib==19.24.4 face_recognition==1.3.0 opencv-python
```