#pragma once

#include <ros/ros.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace calibration_data {

class camera_t {
   public:
    cv::Mat distCoeffs;
    cv::Mat intrisicMat;
    cv::Mat extrinsicMat_RT;
    cv::Mat extrinsicMat_R;
    cv::Mat extrinsicMat_T;
    // Eigen rotation matrix
    Eigen::Matrix3d extrinsicMat_R_eigen;
    // Eigen translation matrix
    Eigen::Vector3d extrinsicMat_T_eigen;
    double fx, fy, cx, cy;

    camera_t() {
        distCoeffs = cv::Mat(5, 1, cv::DataType<double>::type);
        intrisicMat = cv::Mat(3, 3, cv::DataType<double>::type);
        extrinsicMat_RT = cv::Mat(4, 4, cv::DataType<double>::type);  // 外参旋转矩阵3*3和平移向量3*1
        extrinsicMat_R = cv::Mat(3, 3, cv::DataType<double>::type);
        extrinsicMat_T = cv::Mat(3, 1, cv::DataType<double>::type);
    }

    void init(ros::NodeHandle& nh) {
        std::vector<double> extrinsicRT;
        std::vector<double> intrinsic;
        std::vector<double> distCoeff;
        nh.param<std::vector<double>>("camera/extrinsic_RT", extrinsicRT, std::vector<double>());
        nh.param<std::vector<double>>("camera/intrinsic", intrinsic, std::vector<double>());
        nh.param<std::vector<double>>("camera/distCoeff", distCoeff, std::vector<double>());

        this->load_camera_params(extrinsicRT, intrinsic, distCoeff);
    }

    void load_camera_params(std::vector<double> extrinsicRT, std::vector<double> intrinsic, std::vector<double> distCoeff) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                this->extrinsicMat_RT.at<double>(i, j) = extrinsicRT[4 * i + j];
                // std::cout<<extrinsicMat_RT.at<double>(i, j)<<std::endl;
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                this->intrisicMat.at<double>(i, j) = intrinsic[3 * i + j];
                // std::cout<<intrisicMat.at<double>(i, j)<<std::endl;
            }
        }
        for (int i = 0; i < 5; i++) {
            this->distCoeffs.at<double>(i) = distCoeff[i];
            // std::cout<<distCoeffs.at<double>(i)<<std::endl;
        }
        this->fx = intrisicMat.at<double>(0, 0);
        this->fy = intrisicMat.at<double>(1, 1);
        this->cx = intrisicMat.at<double>(0, 2);
        this->cy = intrisicMat.at<double>(1, 2);
        // 3x3 rotation matrix from 4x4 extrinsic matrix
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                this->extrinsicMat_R.at<double>(i, j) = this->extrinsicMat_RT.at<double>(i, j);
            }
        }
        // 3*3 eigen matrix from 3*3 opencv matrix
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                this->extrinsicMat_R_eigen(i, j) = this->extrinsicMat_R.at<double>(i, j);
            }
        }
        // 3x1 translation matrix from 4x4 extrinsic matrix
        for (int i = 0; i < 3; i++) {
            this->extrinsicMat_T.at<double>(i) = this->extrinsicMat_RT.at<double>(i, 3);
        }
        // 3*1 eigen matrix from 3*1 opencv matrix
        for (int i = 0; i < 3; i++) {
            this->extrinsicMat_T_eigen(i) = this->extrinsicMat_T.at<double>(i);
        }
    }
};

// global camera variable, make this a vector later for multiple camera support
camera_t camera;

}  // namespace calibration_data