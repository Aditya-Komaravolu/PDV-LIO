#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>

#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <unordered_map>

#include "../calibration_data.hpp"
#include "../loguru.hpp"
#include "../r3live_coloring.hpp"

struct pose_t {
    pose_t(Eigen::Quaterniond _q = Eigen::Quaterniond(1, 0, 0, 0), Eigen::Vector3d _t = Eigen::Vector3d(0, 0, 0)) : q(_q), t(_t) {}

    Eigen::Quaterniond q;
    Eigen::Vector3d t;
};

void color_pointcloud(cv::Mat& img, pcl::PointCloud<pcl::PointXYZINormal>& point_cloud, pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud) {
    double scale = 1;

    // pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;

    Eigen::Matrix3d rot_mat = calibration_data::camera.extrinsicMat_R_eigen;
    Eigen::Vector3d trans_vec = calibration_data::camera.extrinsicMat_T_eigen;

    // LOG_S(INFO) << "rot_mat: " << rot_mat << std::endl;
    // LOG_S(INFO) << "trans_vec: " << trans_vec << std::endl;
    // LOG_S(INFO) << "fx fy cx cy: " << calibration_data::camera.fx << " " << calibration_data::camera.fy << " " << calibration_data::camera.cx << " " << calibration_data::camera.cy << std::endl;

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
        colored_pt.r = rgb[2];
        colored_pt.g = rgb[1];
        colored_pt.b = rgb[0];

        colored_cloud.push_back(colored_pt);
    }

    // LOG_S(INFO) << "saving colored pcd" << std::endl;
    // pcl::io::savePCDFileBinary("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/colored.pcd", colored_cloud);
}

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "pc_segment");

    ros::NodeHandle nh;

    calibration_data::camera.init(nh);

    LOG_S(INFO) << "running point cloud segmentation..." << std::endl;

    bool color_point_cloud = false;

    std::unordered_map<std::uint16_t, std::string> frame_idx_to_image;

    if (color_point_cloud) {
        for (auto& p : boost::filesystem::recursive_directory_iterator("/home/inkers/satyajit/catkin_pv_lio/segmentation_test2/image_pose/")) {
            if (p.path().extension() == std::string(".png")) {
                // LOG_S(INFO) << p.path() << std::endl;

                std::string filename = p.path().stem().string();

                std::vector<std::string> split_sentence;

                boost::split(split_sentence, filename, boost::is_any_of("_"));

                if (split_sentence.size() != 4) {
                    throw new std::runtime_error("invalid filename " + filename + " should be something like fi_900_ci_903");
                }

                uint16_t frame_index = std::stoi(split_sentence.at(1));

                frame_idx_to_image[frame_index] = p.path().string();
            }
        }

        // for (auto& [idx, img_path] : frame_idx_to_image) {
        //     LOG_S(INFO) << "idx: " << idx << " , path: " << img_path << std::endl;
        // }
    }

    // return EXIT_SUCCESS;

    // read hba poses
    std::vector<pose_t> pose_vec;
    // std::string pose_file = "/home/inkers/satyajit/catkin_pv_lio/segmentation_test2/hba_dump/pose.json";
    std::string pose_file = "/home/inkers/satyajit/dlf_gf_hba_dump/pose.json";
    {
        std::fstream file;
        file.open(pose_file);
        double tx, ty, tz, qw, qx, qy, qz;
        while (!file.eof()) {
            file >> tx >> ty >> tz >> qw >> qx >> qy >> qz;
            Eigen::Quaterniond q(qw, qx, qy, qz);
            Eigen::Vector3d t(tx, ty, tz);
            pose_vec.push_back(pose_t(q, t));
        }
        file.close();
    }

    LOG_S(INFO) << "pose size: " << pose_vec.size() << std::endl;

    std::map<std::uint16_t, std::tuple<pcl::PointCloud<pcl::PointXYZINormal>, pose_t>>
        hba_pcds;

    // std::string hba_pcd_base_path = "/home/inkers/satyajit/catkin_pv_lio/segmentation_test2/hba_dump/pcd/";
    std::string hba_pcd_base_path = "/home/inkers/satyajit/dlf_gf_hba_dump/pcd/";

    pcl::PointCloud<pcl::PointXYZINormal> full_pc;
    pcl::PointCloud<pcl::PointXYZRGB> full_colored_pc;

    for (size_t i = 0; i < pose_vec.size() - 1; i++) {
        LOG_S(INFO) << "processed: "
                    << "[" << i << "/" << pose_vec.size() - 1 << "]" << std::endl;
        pcl::PointCloud<pcl::PointXYZINormal> point_cloud;
        // TODO: i+1 is done only because the modified pv lio we are incrementing frame index first and then dumping
        // pcl::io::loadPCDFile(hba_pcd_base_path + std::to_string(i + 1) + ".pcd", point_cloud);

        pcl::io::loadPCDFile(hba_pcd_base_path + std::to_string(i) + ".pcd", point_cloud);

        // color bhi yahin kar lete hain

        pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;

        if (color_point_cloud) {
            if (frame_idx_to_image.find(i + 1) != frame_idx_to_image.end()) {
                // found frame index in image
                cv::Mat image = cv::imread(frame_idx_to_image[i + 1]);
                color_pointcloud(image, point_cloud, colored_cloud);
            } else {
                LOG_S(WARNING) << "idx: " << i + 1 << " not found" << std::endl;
            }
        }

        if (colored_cloud.size() > 0) {
            for (auto& point : colored_cloud) {
                Eigen::Vector3d new_point(point.x, point.y, point.z);
                new_point = pose_vec.at(i).q * new_point + pose_vec.at(i).t;

                point.x = new_point.x();
                point.y = new_point.y();
                point.z = new_point.z();
            }
            full_colored_pc += colored_cloud;
        }

        if (point_cloud.size() > 0) {
            for (auto& point : point_cloud) {
                Eigen::Vector3d new_point(point.x, point.y, point.z);
                new_point = pose_vec.at(i).q * new_point + pose_vec.at(i).t;

                point.x = new_point.x();
                point.y = new_point.y();
                point.z = new_point.z();
            }
            full_pc += point_cloud;
        }
    }

    LOG_S(INFO) << "saving full pcd" << std::endl;
    // pcl::io::savePCDFileBinary("/home/inkers/satyajit/catkin_pv_lio/segmentation_test2/full_cloud.pcd", full_pc);
    // pcl::io::savePCDFileBinary("/home/inkers/satyajit/catkin_pv_lio/segmentation_test2/full_colored_cloud.pcd", full_colored_pc);
    pcl::io::savePCDFileBinary("/home/inkers/satyajit/dlf_gf_hba_dump/full_cloud.pcd", full_pc);

    // for (auto& p : std::filesystem::recursive_directory_iterator("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/hba_dump/pcd/")) {
    //     if (p.path().extension() == std::string(".pcd")) {
    //         pcl::PointCloud<pcl::PointXYZINormal> point_cloud;
    //         pcl::io::loadPCDFile(p.path(), point_cloud);

    //         hba_pcds[std::stoi(p.path().stem())] = point_cloud;

    //         // cv::Mat image = cv::imread("/home/inkers/satyajit/catkin_pv_lio/segmentation_test/image_pose/fi_900_ci_903.png");

    //         // // LOG_S(INFO) << "image " << image.size << ", point cloud: " << point_cloud.size() << std::endl;
    //         // pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;
    //         // color_pointcloud(image, point_cloud, colored_cloud);
    //     }
    // }

    return EXIT_SUCCESS;
}