#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>

#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "../calibration_data.hpp"
#include "../loguru.hpp"
#include "../r3live_coloring.hpp"

void color_pointcloud(cv::Mat& img, pcl::PointCloud<pcl::PointXYZINormal>& point_cloud) {
    double scale = 1;

    pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;

    Eigen::Matrix3d rot_mat = calibration_data::camera.extrinsicMat_R_eigen;
    Eigen::Vector3d trans_vec = calibration_data::camera.extrinsicMat_T_eigen;

    LOG_S(INFO) << "rot_mat: " << rot_mat << std::endl;
    LOG_S(INFO) << "trans_vec: " << trans_vec << std::endl;
    LOG_S(INFO) << "fx fy cx cy: " << calibration_data::camera.fx << " " << calibration_data::camera.fy << " " << calibration_data::camera.cx << " " << calibration_data::camera.cy << std::endl;

    for (auto& point : point_cloud) {
        // LOG_S(INFO) << "point [" << point.x << ", " << point.y << ", " << point.z << " ]" << std::endl;
        Eigen::Vector3d point_pos(point.x, point.y, point.z);

        Eigen::Vector3d point_in_cam = rot_mat * point_pos + trans_vec;

        if (point_in_cam.z() < 0.001) {
            continue;
        }

        double u = (point_in_cam.x() * calibration_data::camera.fx / point_in_cam.z() + calibration_data::camera.cx) * scale;
        double v = (point_in_cam.y() * calibration_data::camera.fy / point_in_cam.z() + calibration_data::camera.cy) * scale;

        if (((u / scale >= (coloring_config::used_fov_margin * img.cols + 1)) && (std::ceil(u / scale) < ((1 - coloring_config::used_fov_margin) * img.cols)) &&
             (v / scale >= (coloring_config::used_fov_margin * img.rows + 1)) && (std::ceil(v / scale) < ((1 - coloring_config::used_fov_margin) * img.rows)))) {
        } else {
            continue;
        }

        cv::Vec3b rgb = getSubPixel<cv::Vec3b>(img, v, u, 0);

        pcl::PointXYZRGB colored_pt;
        colored_pt.x = point.x;
        colored_pt.y = point.y;
        colored_pt.z = point.z;
        colored_pt.r = rgb[0];
        colored_pt.g = rgb[1];
        colored_pt.b = rgb[2];

        colored_cloud.push_back(colored_pt);
    }

    LOG_S(INFO) << "saving colored pcd" << std::endl;
    pcl::io::savePCDFileBinary("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/colored.pcd", colored_cloud);
}

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "pc_segment");

    ros::NodeHandle nh;

    calibration_data::camera.init(nh);

    LOG_S(INFO) << "running point cloud segmentation..." << std::endl;

    // cv::Mat image = cv::imread("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/image_pose/fi_900_ci_903.png");
    cv::Mat image = cv::imread("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/test.png");

    pcl::PointCloud<pcl::PointXYZINormal> merged_cloud;

    pcl::PointCloud<pcl::PointXYZINormal> point_cloud;
    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/900.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/901.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/902.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/903.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/904.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/905.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/906.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/907.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/908.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/909.pcd", point_cloud);
    merged_cloud += point_cloud;

    pcl::io::loadPCDFile("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/910.pcd", point_cloud);
    merged_cloud += point_cloud;

    // LOG_S(INFO) << "image " << image.size << ", point cloud: " << point_cloud.size() << std::endl;

    color_pointcloud(image, merged_cloud);

    return EXIT_SUCCESS;
}