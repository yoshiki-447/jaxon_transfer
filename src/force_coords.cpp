#include <ros/ros.h>
#include <geometry_msgs/WrenchStamped.h>
#include <tf/transform_listener.h>

class ForceTransformer {
public:
  ForceTransformer(){
    rh_force_sub_ = nh_.subscribe("/off_rhsensor", 1, &ForceTransformer::rh_forceCallback, this);
    rh_force_pub_ = nh_.advertise<geometry_msgs::WrenchStamped>("/off_world_rhsensor", 1);

    lh_force_sub_ = nh_.subscribe("/off_lhsensor", 1, &ForceTransformer::lh_forceCallback, this);
    lh_force_pub_ = nh_.advertise<geometry_msgs::WrenchStamped>("/off_world_lhsensor", 1);
  }
private:
  ros::NodeHandle nh_;
  ros::Subscriber rh_force_sub_;
  ros::Publisher rh_force_pub_;
  tf::TransformListener rh_tf_listener_;
  ros::Subscriber lh_force_sub_;
  ros::Publisher lh_force_pub_;
  tf::TransformListener lh_tf_listener_;
  void rh_forceCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg) {
    try{
      tf::StampedTransform rh_transform;
      rh_tf_listener_.lookupTransform("BODY", msg->header.frame_id, ros::Time(0), rh_transform);  // msg->header.stamp,ros::Time(0)

      geometry_msgs::WrenchStamped rh_transformed_wrench;
      rh_transformed_wrench.header.stamp = msg->header.stamp;
      rh_transformed_wrench.header.frame_id = "BODY";

      tf::Vector3 rh_force(msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z);
      tf::Vector3 rh_torque(msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z);

      tf::Vector3 rh_transformed_force = rh_transform.getBasis() * rh_force;
      rh_transformed_wrench.wrench.force.x = rh_transformed_force.x();
      rh_transformed_wrench.wrench.force.y = rh_transformed_force.y();
      rh_transformed_wrench.wrench.force.z = rh_transformed_force.z();

      tf::Vector3 rh_transformed_torque = rh_transform.getBasis() * rh_torque;
      rh_transformed_wrench.wrench.torque.x = rh_transformed_torque.x();
      rh_transformed_wrench.wrench.torque.y = rh_transformed_torque.y();
      rh_transformed_wrench.wrench.torque.z = rh_transformed_torque.z();

      rh_force_pub_.publish(rh_transformed_wrench); 
    } catch (tf::TransformException& ex) {
      ROS_WARN("Transform error: %s", ex.what());
    }
  }

 void lh_forceCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg) {
    try{
      tf::StampedTransform lh_transform;
      lh_tf_listener_.lookupTransform("BODY", msg->header.frame_id, ros::Time(0), lh_transform);

      geometry_msgs::WrenchStamped lh_transformed_wrench;
      lh_transformed_wrench.header.stamp = msg->header.stamp;
      lh_transformed_wrench.header.frame_id = "BODY";

      tf::Vector3 lh_force(msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z);
      tf::Vector3 lh_torque(msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z);

      tf::Vector3 lh_transformed_force = lh_transform.getBasis() * lh_force;
      lh_transformed_wrench.wrench.force.x = lh_transformed_force.x();
      lh_transformed_wrench.wrench.force.y = lh_transformed_force.y();
      lh_transformed_wrench.wrench.force.z = lh_transformed_force.z();

      tf::Vector3 lh_transformed_torque = lh_transform.getBasis() * lh_torque;
      lh_transformed_wrench.wrench.torque.x = lh_transformed_torque.x();
      lh_transformed_wrench.wrench.torque.y = lh_transformed_torque.y();
      lh_transformed_wrench.wrench.torque.z = lh_transformed_torque.z();

      lh_force_pub_.publish(lh_transformed_wrench); 
    } catch (tf::TransformException& ex) {
      ROS_WARN("Transform error: %s", ex.what());
    }
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "force_transformer_node");
  ForceTransformer force_transformer;
  ros::spin();
  return 0;
}
