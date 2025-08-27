#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

using namespace std::chrono_literals;

class UsbCameraNode : public rclcpp::Node
{
public:
  explicit UsbCameraNode(const rclcpp::NodeOptions & options)
  : Node("usb_camera_node", options)
  {
    // 从参数服务器声明并获取参数
    int device_id = this->declare_parameter<int>("device_id", 0);
    std::string image_topic = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    int publish_fps = this->declare_parameter<int>("publish_fps", 30);

    // 打开摄像头
    cap_.open(device_id, cv::CAP_ANY);
    if (!cap_.isOpened()) {
      RCLCPP_ERROR(get_logger(), "无法打开摄像头 device_id=%d", device_id);
      rclcpp::shutdown();
      return;
    }
    RCLCPP_INFO(get_logger(), "打开摄像头 device_id=%d, 发布话题=%s, 帧率=%d",
                device_id, image_topic.c_str(), publish_fps);

    // Publisher
    pub_ = this->create_publisher<sensor_msgs::msg::Image>(image_topic, 10);

    // 定时器
    auto period = std::chrono::milliseconds(1000 / publish_fps);
    timer_ = this->create_wall_timer(period,
      std::bind(&UsbCameraNode::on_timer, this));
  }

private:
  void on_timer()
  {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_WARN(get_logger(), "采集不到帧，正在重置...");
      cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
      return;
    }
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame)
                 .toImageMsg();
    msg->header.stamp = now();
    pub_->publish(*msg);
  }

  cv::VideoCapture cap_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

static bool has_params_file_arg(int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i] ? argv[i] : "";
    if (a == "--params-file") return true;                 // 形式：--params-file <path>
    if (a.rfind("--params-file=", 0) == 0) return true;    // 形式：--params-file=<path>
  }
  return false;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // 你的默认参数文件（不存在就不加）
  const std::string default_yaml =
      "/home/unitree/ros2_ws/LeggedRobot/src/Ros2ImageProcess/config.yaml";

  rclcpp::NodeOptions options;

  // 只有当命令行没有带 --params-file 时，才自动加默认 YAML
  if (!has_params_file_arg(argc, argv)) {
    if (std::filesystem::exists(default_yaml)) {
      options.arguments({"--ros-args", "--params-file", default_yaml});
      // 可选：打印一下提示
      RCLCPP_INFO(rclcpp::get_logger("usb_camera_node"),
                  "未检测到 --params-file，使用默认参数文件：%s",
                  default_yaml.c_str());
    } else {
      RCLCPP_WARN(rclcpp::get_logger("usb_camera_node"),
                  "未检测到 --params-file，且默认参数文件不存在：%s，改用节点内默认参数。",
                  default_yaml.c_str());
    }
  }

  auto node = std::make_shared<UsbCameraNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}