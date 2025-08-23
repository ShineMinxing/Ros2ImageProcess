#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <yolo_msgs/msg/detection_array.hpp>

class DroneDetectorNode : public rclcpp::Node
{
public:
  explicit DroneDetectorNode(const rclcpp::NodeOptions & opts)
  : Node("drone_detector_node", opts)
  {
    /* ---------- 参数 ---------- */
    det_topic_    = declare_parameter<std::string>("tracking_topic",  "/yolo/tracking");
    cmd_topic_    = declare_parameter<std::string>("cmd_topic",       "SMX/SportCmd");
    angle_topic_  = declare_parameter<std::string>("angle_topic",     "SMX/TargetImageAngle");
    img_width_    = declare_parameter<int>("image_width",             1280);   // 720p
    img_height_   = declare_parameter<int>("image_height",            720);
    fov_h_deg_    = declare_parameter<double>("fov_h_deg",            125.0);  // =√(143²-69²)≈125
    fov_v_deg_    = declare_parameter<double>("fov_v_deg",             69.0);

    /* ---------- 通信 ---------- */
    det_sub_  = create_subscription<yolo_msgs::msg::DetectionArray>(
        det_topic_, 10,
        std::bind(&DroneDetectorNode::detCb, this, std::placeholders::_1));

    cmd_sub_  = create_subscription<std_msgs::msg::Float64MultiArray>(
        cmd_topic_, 10,
        std::bind(&DroneDetectorNode::cmdCb, this, std::placeholders::_1));

    angle_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(angle_topic_, 10);

    RCLCPP_INFO(get_logger(),
                "DroneDetectorNode started. det=%s cmd=%s -> angle=%s",
                det_topic_.c_str(), cmd_topic_.c_str(), angle_topic_.c_str());
  }

private:
  /* ======== 指令回调 ======== */
  void cmdCb(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (!msg->data.empty() && static_cast<int>(msg->data[0]) == 22110000)
    {
      tracking_enabled_ = true;
      tracked_id_.clear();   // 重新搜目标
      RCLCPP_INFO(get_logger(), "Auto-Track ENABLE → 寻找 Drone…");
    }
  }

  /* ======== 检测回调 ======== */
  void detCb(const yolo_msgs::msg::DetectionArray::SharedPtr msg)
  {
    if (!tracking_enabled_) return;

    const auto & dets = msg->detections;
    if (dets.empty()) return;

    /* -- 1. 若还未锁定 ID，就选离中心最近的 “Drone” -- */
    if (tracked_id_.empty())
    {
      double best_dist2 = 1e9;
      for (const auto & d : dets)
      {
        if (d.class_name != "drone") continue;
        double dx = d.bbox.center.position.x - img_width_  / 2.0;
        double dy = d.bbox.center.position.y - img_height_ / 2.0;
        double dist2 = dx*dx + dy*dy;
        if (dist2 < best_dist2)
        {
          best_dist2 = dist2;
          tracked_id_ = d.id;
          last_pt_ = {d.bbox.center.position.x, d.bbox.center.position.y};
        }
      }
      if (!tracked_id_.empty())
        RCLCPP_INFO(get_logger(), "Locked on Drone id=%s", tracked_id_.c_str());
    }
    else
    {
      /* -- 2. 追踪已有 ID -- */
      bool found = false;
      for (const auto & d : dets)
      {
        if (d.id == tracked_id_)
        {
          last_pt_ = {d.bbox.center.position.x, d.bbox.center.position.y};
          found = true;
          break;
        }
      }
      if (!found)
      {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Lost id=%s, re-searching…", tracked_id_.c_str());
        tracked_id_.clear();   // 重新搜索
        return;
      }

      /* -- 3. 计算 FOV 角（弧度）并发布 -- */
      double dx = last_pt_.first  - img_width_  / 2.0;
      double dy = -(last_pt_.second - img_height_ / 2.0);          // y 轴向上为正
      double rx = dx / (img_width_  / 2.0);   // -1…1
      double ry = dy / (img_height_ / 2.0);

      double ang_x = rx * (fov_h_deg_/2.0);
      double ang_y = ry * (fov_v_deg_/2.0);

      std_msgs::msg::Float64MultiArray out;
      out.data = {ang_x, ang_y, 0.0};
      angle_pub_->publish(out);
    }
  }

  /* ---------- 成员 ---------- */
  std::string det_topic_, cmd_topic_, angle_topic_;
  int    img_width_, img_height_;
  double fov_h_deg_, fov_v_deg_;

  bool tracking_enabled_{false};
  std::string tracked_id_;
  std::pair<double,double> last_pt_{0,0};

  rclcpp::Subscription<yolo_msgs::msg::DetectionArray>::SharedPtr det_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr    angle_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.allow_undeclared_parameters(true)
     .automatically_declare_parameters_from_overrides(true);
  rclcpp::spin(std::make_shared<DroneDetectorNode>(opt));
  rclcpp::shutdown();
  return 0;
}
