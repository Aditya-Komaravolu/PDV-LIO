#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <boost/foreach.hpp>
#include <memory>
#define foreach BOOST_FOREACH

#include "../calibration_data.hpp"

ros::Publisher odom_pub1;
ros::Publisher odom_pub2;

int total_count = 0;

void odom_callback(const nav_msgs::Odometry odom) {
    // std::cout << odom.pose << std::endl;

    auto position = odom.pose.pose.position;
    auto orientation = odom.pose.pose.orientation;

    std::cout << "position: " << position << std::endl;
    std::cout << "orientation: " << orientation << std::endl;

    Eigen::Vector3d cur_pos(position.x, position.y, position.z);
    Eigen::Quaterniond cur_quat(orientation.w, orientation.x, orientation.y, orientation.z);

    auto rot_mat = cur_quat.toRotationMatrix();

    auto rot_wrt_cam = rot_mat * calibration_data::camera.extrinsicMat_R_eigen.inverse();
    auto trans_wrt_cam = rot_mat * calibration_data::camera.extrinsicMat_T_eigen + cur_pos;

    Eigen::Quaterniond q;
    q = Eigen::Quaterniond(rot_wrt_cam);

    std::string txt_file_name = "/home/inkers/satyajit/catkin_pv_lio/pose.txt";

    FILE* fp = fopen(txt_file_name.c_str(), "a+");
    if (fp) {
        fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf\r\n", q.w(), q.x(), q.y(), q.z(),
                trans_wrt_cam(0), trans_wrt_cam(1), trans_wrt_cam(2));
        fclose(fp);
    }

    ros::Time current_time = ros::Time::now();
    // {
    //     tf::TransformBroadcaster odom_broadcaster;
    //     geometry_msgs::Quaternion odom_quat;
    //     odom_quat.w = 1;
    //     odom_quat.x = 0;
    //     odom_quat.y = 0;
    //     odom_quat.z = 0;
    //     geometry_msgs::TransformStamped odom_trans;
    //     odom_trans.header.stamp = current_time;

    //     odom_trans.header.frame_id = "odom";
    //     odom_trans.child_frame_id = "base_link";
    //     odom_trans.transform.translation.x = 0;
    //     odom_trans.transform.translation.y = 0;
    //     odom_trans.transform.translation.z = 0;
    //     odom_trans.transform.rotation = odom_quat;
    //     odom_broadcaster.sendTransform(odom_trans);
    // }
    {
        nav_msgs::Odometry odom;
        odom.header.stamp = current_time;
        odom.header.frame_id = "odom";
        odom.pose.pose.position = position;
        odom.pose.pose.orientation = orientation;
        odom.child_frame_id = "base_link";
        odom_pub1.publish(odom);
    }
    {
        nav_msgs::Odometry odom;
        odom.header.stamp = current_time;
        odom.header.frame_id = "odom";
        odom.pose.pose.position.x = trans_wrt_cam.x();
        odom.pose.pose.position.y = trans_wrt_cam.y();
        odom.pose.pose.position.z = trans_wrt_cam.z();
        odom.pose.pose.orientation.w = q.w();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.child_frame_id = "base_link";
        odom_pub2.publish(odom);
    }

    total_count++;

    std::cout << "total_count: " << total_count << std::endl;
}

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "lodom_to_codom");
    ros::NodeHandle nh;

    calibration_data::camera.init(nh);

    // odom_pub1 = nh.advertise<nav_msgs::Odometry>("odom_orig", 5000);
    // odom_pub2 = nh.advertise<nav_msgs::Odometry>("odom_transformed", 5000);

    // ros::Subscriber odomSub = nh.subscribe("/Odometry", 100000, odom_callback);

    std::string txt_file_name = "/home/inkers/satyajit/catkin_pv_lio/pose.txt";

    FILE* fp = fopen(txt_file_name.c_str(), "w");
    fclose(fp);

    rosbag::Bag bag("/home/inkers/rgb_odometry.bag");

    rosbag::View view(bag, rosbag::TopicQuery("/Odometry"));

    int counter = 0;

    foreach (rosbag::MessageInstance const m, view) {
        std::cout << ++counter << std::endl;
        nav_msgs::Odometry::ConstPtr odom = m.instantiate<nav_msgs::Odometry>();

        auto position = odom->pose.pose.position;
        auto orientation = odom->pose.pose.orientation;

        std::cout << "position: " << position << std::endl;
        std::cout << "orientation: " << orientation << std::endl;
        Eigen::Vector3d cur_pos(position.x, position.y, position.z);
        Eigen::Quaterniond cur_quat(orientation.w, orientation.x, orientation.y, orientation.z);

        auto rot_mat = cur_quat.toRotationMatrix();

        auto rot_wrt_cam = rot_mat * calibration_data::camera.extrinsicMat_R_eigen.inverse();
        auto trans_wrt_cam = rot_mat * calibration_data::camera.extrinsicMat_T_eigen + cur_pos;

        Eigen::Quaterniond q;
        q = Eigen::Quaterniond(rot_wrt_cam);

        std::string txt_file_name = "/home/inkers/satyajit/catkin_pv_lio/pose.txt";

        FILE* fp = fopen(txt_file_name.c_str(), "a+");
        if (fp) {
            fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf\r\n", q.w(), q.x(), q.y(), q.z(),
                    trans_wrt_cam(0), trans_wrt_cam(1), trans_wrt_cam(2));
            fclose(fp);
        }
    }

    // ros::spin();

    return EXIT_SUCCESS;
}