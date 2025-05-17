#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std::chrono_literals;

class VideoPlayNode : public rclcpp::Node
{
public:
  explicit VideoPlayNode(const rclcpp::NodeOptions & options)
  : Node("video_play_node", options)
  {
    // 声明并获取参数
    std::string video_path = this->declare_parameter<std::string>(
      "VIDEO_FILE_PATH", "/home/smx/NetworkShare/Video1_4.mp4");
    std::string image_topic = this->declare_parameter<std::string>(
      "IMAGE_TOPIC", "NoYamlRead/GimbalCamera");
    int publish_fps = this->declare_parameter<int>(
      "PUBLISH_FPS", 30);

    // 创建 Publisher
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      image_topic, 10);

    // 打开视频文件
    cap_.open(video_path, cv::CAP_ANY);
    if (!cap_.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "无法打开视频文件: %s", video_path.c_str());
      return;
    }
    RCLCPP_INFO(this->get_logger(), "成功打开视频: %s", video_path.c_str());

    // 计算发布周期
    auto period = std::chrono::milliseconds(1000 / publish_fps);
    timer_ = this->create_wall_timer(
      period, std::bind(&VideoPlayNode::timerCallback, this));
  }

private:
  void timerCallback()
  {
    if (!cap_.isOpened()) return;

    cv::Mat frame;
    cap_ >> frame;
    if (frame.empty()) {
      cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
      cap_ >> frame;
      if (frame.empty()) {
        RCLCPP_WARN(this->get_logger(), "读取视频帧为空！");
        return;
      }
    }

    auto msg = cv_bridge::CvImage(
      std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = this->now();
    msg->header.frame_id = "video_play";
    publisher_->publish(*msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  cv::VideoCapture cap_;
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
  auto node = std::make_shared<VideoPlayNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}