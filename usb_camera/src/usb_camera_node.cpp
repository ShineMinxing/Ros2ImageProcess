#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <chrono>

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

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.arguments({
    "--ros-args",
    "--params-file",
    "/home/unitree/ros2_ws/LeggedRobot/src/Ros2ImageProcess/config.yaml"
  });

  auto node = std::make_shared<UsbCameraNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
