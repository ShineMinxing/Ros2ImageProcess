#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;

class SpotDetectorNode : public rclcpp::Node
{
public:
  explicit SpotDetectorNode(const rclcpp::NodeOptions & options)
  : Node("spot_detector_node", options)
  {
    // 声明并获取参数
    input_topic_ = this->declare_parameter<std::string>(
      "IMAGE_INPUT_TOPIC", "/SMX/Go2Camera");
    output_image_topic_ = this->declare_parameter<std::string>(
      "IMAGE_OUTPUT_TOPIC", "/SMX/TargetImage");
    output_angle_topic_ = this->declare_parameter<std::string>(
      "ANGLE_OUTPUT_TOPIC", "/SMX/TargetImageAngle");
    fov_h_ = this->declare_parameter<double>("FOV_H", 125.0);
    fov_v_ = this->declare_parameter<double>("FOV_V",  69.0);

    // 订阅图像
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      input_topic_, 10,
      std::bind(&SpotDetectorNode::imageCallback, this, std::placeholders::_1));

    // 发布处理后图像
    video_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      output_image_topic_, 10);

    // 发布角度信息
    angle_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      output_angle_topic_, 10);

    RCLCPP_INFO(this->get_logger(),
      "SpotDetectorNode 启动： in=%s out_img=%s out_ang=%s FOV=(%.1f,%.1f)",
      input_topic_.c_str(), output_image_topic_.c_str(),
      output_angle_topic_.c_str(), fov_h_, fov_v_);
  }

private:
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    cv::Mat frame = cv_ptr->image;
    int width = frame.cols, height = frame.rows;
    if (frame.empty()) return;

    bool found = false;
    int max_g = 0, mx=0, my=0;
    for (int y=0; y<height; ++y) {
      const uchar* row = frame.ptr<uchar>(y);
      for (int x=0; x<width; ++x) {
        int b=row[x*3], g=row[x*3+1], r=row[x*3+2];
        if (g>200 && g>(r+50) && g>(b+50) && g>max_g) {
          max_g=g; mx=x; my=y; found=true;
        }
      }
    }

    if (found) {
      cv::circle(frame, {mx,my}, 10, {0,255,0}, 2);
      double dx = mx - width/2.0;
      double dy = -(my - height/2.0);
      double rx = dx / (width/2.0);
      double ry = dy / (height/2.0);
      double angle_x = rx * (fov_h_/2.0);
      double angle_y = ry * (fov_v_/2.0);

      // 发布图像
      auto out_img = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
      video_pub_->publish(*out_img);

      // 发布角度
      std_msgs::msg::Float64MultiArray ang;
      ang.data = {angle_x, angle_y, 0.0};
      angle_pub_->publish(ang);
    }
  }

  // 参数
  std::string input_topic_, output_image_topic_, output_angle_topic_;
  double fov_h_, fov_v_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr video_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr angle_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opts;
  opts.arguments({
    "--ros-args",
    "--params-file",
    "src/Ros2ImageProcess/config.yaml"
  });
  auto node = std::make_shared<SpotDetectorNode>(opts);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}