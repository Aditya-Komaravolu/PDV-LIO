// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#define PCL_NO_PRECOMPILE  // !! BEFORE ANY PCL INCLUDE!!

#include <Python.h>
#include <geometry_msgs/Vector3.h>
#include <image_transport/image_transport.h>
#include <livox_ros_driver/CustomMsg.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <so3_math.h>
#include <std_srvs/SetBool.h>
#include <std_srvs/Trigger.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <unistd.h>

#include <Eigen/Core>
#include <csignal>
#include <experimental/filesystem>
#include <fstream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "IMU_Processing.hpp"
#include "calibration_data.hpp"
#include "common.hpp"
#include "loguru.hpp"
#include "openmvs_utils.hpp"
#include "preprocess.h"
#include "r3live_coloring.hpp"
#include "services.hpp"
#include "voxel_map_util.hpp"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

namespace filesystem = std::experimental::filesystem;

/**
 * OpenMVS Dump Settings
 */
PointCloudXYZDump point_cloud_dump;

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool time_sync_en = false, extrinsic_est_en = true, path_en = true, pcd_save_en = false;
bool save_openmvs_file_en, save_image_and_pose_en;
bool save_cloud_pcd_en, save_cloud_ply_en, save_cloud_lio_pcd;
double lidar_time_offset = 0.0;
/**************************/

bool save_hba_pose_en = false;
std::string save_base_path;
double time_diff_lidar_to_imu = 0.0;

string map_file_path, lid_topic, imu_topic, cam_topic, pcd_save_path;
double last_timestamp_lidar = 0, last_cam_timestamp = 0, last_timestamp_imu = -1.0;
double last_imu_processed_time = -1.0;

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
int count_ = 0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;

double res_mean_last = 0.05, total_residual = 0.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_index = 0, pcd_save_interval = -1;
bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool kill_on_finish = false;

vector<vector<int>> pointSearchInd_surf;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
std::deque<double> time_buffer;
std::deque<PointCloudXYZINormal::Ptr> lidar_buffer;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
std::deque<sensor_msgs::ImageConstPtr> cam_buffer;

PointCloudXYZINormal::Ptr featsFromMap(new PointCloudXYZINormal());
PointCloudXYZINormal::Ptr feats_undistort(new PointCloudXYZINormal());
PointCloudXYZINormal::Ptr feats_down_body(new PointCloudXYZINormal());
PointCloudXYZINormal::Ptr feats_down_world(new PointCloudXYZINormal());
PointCloudXYZINormal::Ptr normvec(new PointCloudXYZINormal(100000, 1));
// PointCloudXYZRGBI::Ptr laserCloudOri(new PointCloudXYZRGBI(100000, 1));
// PointCloudXYZRGBI::Ptr corr_normvect(new PointCloudXYZRGBI(100000, 1));
PointCloudXYZINormal::Ptr _featsArray;
std::vector<M3D> var_down_body;

pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilterSurf;

std::vector<float> nn_dist_in_feats;
std::vector<float> nn_plane_std;
PointCloudXYZINormal::Ptr feats_with_correspondence(new PointCloudXYZINormal());

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
double sigma_num = 2.0;
double max_voxel_size = 1.0;
std::vector<int> layer_size;

double ranging_cov = 0.0;
double angle_cov = 0.0;
std::vector<double> layer_point_size;

bool publish_voxel_map = false;
bool publish_color_map = false;
int publish_max_voxel_layer = 0;

std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig) {
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

const bool var_contrast(pointWithCov &x, pointWithCov &y) {
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->rgba = pi->rgba;
}

void pointBodyToWorld_ikfom(pcl::PointXYZINormal const *const pi, pcl::PointXYZINormal *const po, state_ikfom &s) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->rgba = pi->rgba;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po) {
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->rgba = pi->rgba;
}

void RGBpointBodyToWorld(pcl::PointXYZINormal const *const pi, pcl::PointXYZINormal *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// void RGBpointBodyToWorld(pcl::PointXYZRGBINormal const *const pi, pcl::PointXYZRGBINormal *const po) {
//     V3D p_body(pi->x, pi->y, pi->z);
//     V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

//     po->x = p_global(0);
//     po->y = p_global(1);
//     po->z = p_global(2);
//     po->intensity = pi->intensity;
// }

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
    po->rgba = pi->rgba;
    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
}

void RGBpointBodyLidarToIMU(pcl::PointXYZINormal const *const pi, pcl::PointXYZINormal *const po) {
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    auto time_offset = lidar_time_offset;
    //    std::printf("lidar offset:%f\n", lidar_time_offset);
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZINormal::Ptr ptr(new PointCloudXYZINormal());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty()) {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty()) {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZINormal::Ptr ptr(new PointCloudXYZINormal());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void rgb_cam_cbk(const sensor_msgs::ImageConstPtr &msg) {
    mtx_buffer.lock();
    if (msg->header.stamp.toSec() < last_cam_timestamp) {
        ROS_ERROR("RGB error loop back, clear buffer");
        cam_buffer.clear();
    }
    last_cam_timestamp = msg->header.stamp.toSec();
    if (!time_sync_en && abs(last_cam_timestamp - last_timestamp_lidar) > 10.0 && !lidar_buffer.empty() && !cam_buffer.empty()) {
        ROS_WARN("Lidar and Rgb not Synced, Lidar time: %lf,RGB header time: %lf \n", last_timestamp_lidar, last_cam_timestamp);
    }
    cam_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu) {
        //        ROS_WARN("imu loop back, clear buffer");
        //        imu_buffer.clear();
        ROS_WARN("imu loop back, ignoring!!!");
        ROS_WARN("current T: %f, last T: %f", timestamp, last_timestamp_imu);
        return;
    }
    // 剔除异常数据
    if (std::abs(msg->angular_velocity.x) > 10 || std::abs(msg->angular_velocity.y) > 10 || std::abs(msg->angular_velocity.z) > 10) {
        ROS_WARN("Large IMU measurement!!! Drop Data!!! %.3f  %.3f  %.3f",
                 msg->angular_velocity.x,
                 msg->angular_velocity.y,
                 msg->angular_velocity.z);
        return;
    }

    //    // 如果是第一帧 拿过来做重力对齐
    //    // TODO 用多帧平均的重力
    //    if (is_first_imu) {
    //        double acc_vec[3] = {msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z};
    //
    //        R__world__o__initial = SO3(g2R(Eigen::Vector3d(acc_vec)));
    //
    //        is_first_imu = false;
    //    }

    last_timestamp_imu = timestamp;
    last_imu_processed_time = ros::Time::now().toSec();

    mtx_buffer.lock();

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
/// @brief syncs lidar, imu and camera
/// @param meas
/// @return is_synced, buffer_empty
std::tuple<bool, bool> sync_packages(MeasureGroup &meas) {
    if (lidar_buffer.empty() || imu_buffer.empty() || cam_buffer.empty()) {
        LOG_S(WARNING) << "QUEUE EMPTY!" << std::endl;
        return std::make_tuple(false, true);
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed) {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1)  // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        } else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        } else {
            //            std::printf("\nFirst 100 points: \n");
            //            for(int i=0; i < 100; ++i){
            //                std::printf("%f ", meas.lidar->points[i].curvature  / double(1000));
            //            }
            //
            //            std::printf("\n Last 100 points: \n");
            //            for(int i=100; i >0; --i){
            //                std::printf("%f ", meas.lidar->points[meas.lidar->size() - i - 1].curvature / double(1000));
            //            }
            //            std::printf("last point offset time: %f\n", meas.lidar->points.back().curvature / double(1000));
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            //            lidar_end_time = meas.lidar_beg_time + (meas.lidar->points[meas.lidar->points.size() - 20]).curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            //            std::printf("pcl_bag_time: %f\n", meas.lidar_beg_time);
            //            std::printf("lidar_end_time: %f\n", lidar_end_time);
        }

        meas.lidar_end_time = lidar_end_time;
        //        std::printf("Scan start timestamp: %f, Scan end time: %f\n", meas.lidar_beg_time, meas.lidar_end_time);

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time) {
        LOG_S(WARNING) << "lidar time > imu time" << std::endl;
        if (common::debug_en) {
            LOG_S(INFO) << "lidar_buffer: " << lidar_buffer.size() << " , imu_buffer: " << imu_buffer.size() << " , cam_buffer: " << cam_buffer.size() << std::endl;
        }
        // if its more than 1 minute since last imu push then mark the queue as finished
        // we'll ignore camera, as we dont get imu, there will be no mapping and camera is useless
        bool imu_queue_finished = (ros::Time::now().toSec() - last_imu_processed_time) > 60;
        std::stringstream stream;
        stream << std::fixed << std::setprecision(10) << "imu_queue not pushed for 60secs, marking imu queue as finished"
               << " last_imu_processed_time: " << last_imu_processed_time << " , current_time: " << ros::Time::now().toSec();

        if (imu_queue_finished) {
            LOG_S(ERROR) << stream.str() << std::endl;
        }
        return std::make_tuple(false, imu_queue_finished);
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    // Pushing RGB cam 1 image
    double cam_time = cam_buffer.front()->header.stamp.toSec();
    meas.cam.clear();
    while ((!cam_buffer.empty()) && (cam_time < lidar_end_time)) {
        // std::lock_guard<std::mutex> lock(mtx_buffer);
        cam_time = cam_buffer.front()->header.stamp.toSec();
        if (cam_time > lidar_end_time) break;
        meas.cam.push_back(cam_buffer.front());
        cam_buffer.pop_front();
    }
    // meas.rgb_cam_1.push_back(rgb_cam_1_buffer.front());
    // rgb_cam_1_buffer.pop_front();
    if (!meas.cam.empty()) {
        if (common::debug_en) {
            LOG_S(INFO) << "rgb frames in queue: " << meas.cam.size() << std::endl;
            LOG_S(INFO) << "current frame index: " << colored_map::colored_frame_index << std::endl;
        }

        auto diff_time = meas.cam.back()->header.stamp.toSec() - time_buffer.front();
        // LOG_S(WARNING) << "diff time" << diff_time << std::endl;
        colored_map::colored_frame_index++;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;

    if (common::debug_en) {
        LOG_S(INFO) << "lidar_buffer: " << lidar_buffer.size() << " , imu_buffer: " << imu_buffer.size() << " , cam_buffer: " << cam_buffer.size() << std::endl;
    }

    return std::make_tuple(true, false);
}

PointCloudXYZINormal::Ptr pcl_wait_pub(new PointCloudXYZINormal(500000, 1));
PointCloudXYZINormal::Ptr pcl_wait_save(new PointCloudXYZINormal());

void publish_frame_world(const ros::Publisher &pubLaserCloudFull) {
    if (scan_pub_en) {
        PointCloudXYZINormal::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZINormal laserCloudWorld;
        for (int i = 0; i < size; i++) {
            pcl::PointXYZINormal const *const p = &laserCloudFullRes->points[i];
            if (p->intensity < 5) {
                continue;
            }
            //            if (p->x < 0 and p->x > -4
            //                    and p->y < 1.5 and p->y > -1.5
            //                            and p->z < 2 and p->z > -1) {
            //                continue;
            //            }
            pcl::PointXYZINormal p_world;

            RGBpointBodyToWorld(p, &p_world);
            //            if (p_world.z > 1) {
            //                continue;
            //            }
            laserCloudWorld.push_back(p_world);
            //            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
//                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (save_cloud_lio_pcd) {
        int size = feats_undistort->points.size();
        PointCloudXYZINormal::Ptr laserCloudWorld(
            new PointCloudXYZINormal(size, 1));

        for (int i = 0; i < size; i++) {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        // static int scan_wait_num = 0;
        // scan_wait_num++;
        // if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
        //     pcd_index++;
        //     string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
        //     pcl::PCDWriter pcd_writer;
        //     cout << "current scan saved to " << all_points_dir << endl;
        //     pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
        //     pcl_wait_save->clear();
        //     scan_wait_num = 0;
        // }
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body) {
    //    int size = feats_undistort->points.size();
    PointCloudXYZINormal::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZINormal::Ptr laserCloudIMUBody(new PointCloudXYZINormal(size, 1));
    for (int i = 0; i < size; i++) {
        RGBpointBodyLidarToIMU(&laserCloudFullRes->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher &pubLaserCloudMap) {
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out) {
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped) {
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);  // ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));

    static tf::TransformBroadcaster br_world;
    transform.setOrigin(tf::Vector3(0, 0, 0));
    q.setValue(p_imu->Initial_R_wrt_G.x(), p_imu->Initial_R_wrt_G.y(), p_imu->Initial_R_wrt_G.z(), p_imu->Initial_R_wrt_G.w());
    transform.setRotation(q);
    br_world.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "camera_init"));
}

void publish_path(const ros::Publisher pubPath) {
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 1 == 0) {
        path.header.stamp = msg_body_pose.header.stamp;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void transformLidar(const state_ikfom &state_point, const PointCloudXYZRGBI::Ptr &input_cloud, PointCloudXYZRGBI::Ptr &trans_cloud) {
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
        pcl::PointXYZRGBINormal p_c = input_cloud->points[i];
        Eigen::Vector3d p_lidar(p_c.x, p_c.y, p_c.z);
        // HACK we need to specify p_body as a V3D type!!!
        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
        PointType pi;
        pi.x = p_body(0);
        pi.y = p_body(1);
        pi.z = p_body(2);
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

void transformLidar(const state_ikfom &state_point, const PointCloudXYZINormal::Ptr &input_cloud, PointCloudXYZINormal::Ptr &trans_cloud) {
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
        pcl::PointXYZINormal p_c = input_cloud->points[i];
        Eigen::Vector3d p_lidar(p_c.x, p_c.y, p_c.z);
        // HACK we need to specify p_body as a V3D type!!!
        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
        pcl::PointXYZINormal pi;
        pi.x = p_body(0);
        pi.y = p_body(1);
        pi.z = p_body(2);
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

// M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom>& kf, const Eigen::Matrix3d& COV_lidar)
//{
//     double match_start = omp_get_wtime();
//     // FIXME 这里首先假定LiDAR系和body是重叠的 没有外参
//     M3D point_crossmat;
//     point_crossmat << SKEW_SYM_MATRX(p_lidar);
//     // 注意这里Rt的cov顺序
//     M3D rot_var = kf.get_P().block<3, 3>(3, 3);
//     M3D t_var = kf.get_P().block<3, 3>(0, 0);
//     auto state = kf.get_x();
//
//     // Eq. (3)
//     M3D COV_world =
//             state.rot * COV_lidar * state.rot.conjugate()
//             + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose()  * state.rot.conjugate()
//             + t_var;
//     return COV_world;
//     // Voxel map 真实实现
////    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
////          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;
//
//}

M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom> &kf, const Eigen::Matrix3d &COV_lidar) {
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(p_lidar);
    auto state = kf.get_x();

    // lidar到body的方差传播
    // 注意外参的var是先rot 后pos
    M3D il_rot_var = kf.get_P().block<3, 3>(6, 6);
    M3D il_t_var = kf.get_P().block<3, 3>(9, 9);

    M3D COV_body =
        state.offset_R_L_I * COV_lidar * state.offset_R_L_I.conjugate() + state.offset_R_L_I * (-point_crossmat) * il_rot_var * (-point_crossmat).transpose() * state.offset_R_L_I.conjugate() + il_t_var;

    // body的坐标
    V3D p_body = state.offset_R_L_I * p_lidar + state.offset_T_L_I;

    // body到world的方差传播
    // 注意pose的var是先pos 后rot
    point_crossmat << SKEW_SYM_MATRX(p_body);
    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
    M3D t_var = kf.get_P().block<3, 3>(0, 0);

    // Eq. (3)
    M3D COV_world =
        state.rot * COV_body * state.rot.conjugate() + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose() * state.rot.conjugate() + t_var;

    return COV_world;
    // Voxel map 真实实现
    //    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
    //          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;
}

void observation_model_share(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    //    laserCloudOri->clear();
    //    corr_normvect->clear();
    feats_with_correspondence->clear();
    total_residual = 0.0;

    // =================================================================================================================
    // 用当前迭代轮最新的位姿估计值 将点云转换到world地图系
    vector<pointWithCov> pv_list;
    PointCloudXYZINormal::Ptr world_lidar(new PointCloudXYZINormal);
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    //    transformLidar(state_point, feats_down_body, world_lidar);
    transformLidar(s, feats_down_body, world_lidar);
    pv_list.resize(feats_down_body->size());
    for (size_t i = 0; i < feats_down_body->size(); i++) {
        // 保存body系和world系坐标
        pointWithCov pv;
        pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
        pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        // 计算lidar点的cov
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
        // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
        M3D cov_lidar = var_down_body[i];
        // 将body系的var转换到world系
        M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);
        pv.cov = cov_world;
        pv.cov_lidar = cov_lidar;
        pv_list[i] = pv;
    }

    // ===============================================================================================================
    // 查找最近点 并构建residual
    double match_start = omp_get_wtime();
    std::vector<ptpl> ptpl_list;
    std::vector<V3D> non_match_list;
    BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list,
                         ptpl_list, non_match_list);
    double match_end = omp_get_wtime();
    // std::printf("Match Time: %f\n", match_end - match_start);

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // 根据匹配结果 设置H和R的维度
    // h_x是观测值对状态量的导数 TODO 为什么不加上状态量对状态量误差的导数？？？？像quaternion那本书？
    effct_feat_num = ptpl_list.size();
    if (effct_feat_num < 1) {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);  // 23 因为点面距离只和位姿 外参有关 对其他状态量的导数都是0
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1);  // 把R作为向量 用的时候转换成diag
//    ekfom_data.R.setZero();
//    printf("isDiag: %d  R norm: %f\n", ekfom_data.R.isDiagonal(1e-10), ekfom_data.R.norm());

//    // 求每个匹配点到平面的距离
//    for (int i = 0; i < ptpl_list.size(); i++) {
//        // 取出匹配到的world系norm
//        PointType pl;
//        pl.x = ptpl_list[i].normal(0);
//        pl.y = ptpl_list[i].normal(1);
//        pl.z = ptpl_list[i].normal(2);
//
//        // 将原始点云转换到world系
//        V3D pi_world(s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos);
//
//        // 求点面距离
//        float dis = pi_world.x() * pl.x + pi_world.y() * pl.y + pi_world.z() * pl.z + ptpl_list[i].d;
//        pl.intensity = dis;
////        std::printf("%.5f   %.5f\n", dis, ptpl_list[i].pd2);
////        std::printf("%.5f  %.5f\n", pi_world.x(), ptpl_list[i].point_world.x());
////        std::printf("%.5f  %.5f\n", pi_world.y(), ptpl_list[i].point_world.y());
//
//        PointType pi_body;
//        pi_body.x = ptpl_list[i].point(0);
//        pi_body.y = ptpl_list[i].point(1);
//        pi_body.z = ptpl_list[i].point(2);
//        laserCloudOri->push_back(pi_body);
//        corr_normvect->push_back(pl);
//        // for visualization
//        feats_with_correspondence->push_back(pi_body);
//
//        total_residual += fabs(dis);
//    }
//    assert(laserCloudOri->size() == effct_feat_num && corr_normvect->size() == effct_feat_num);
// #ifdef MP_EN
//     omp_set_num_threads(MP_PROC_NUM);
// #endif
#pragma omp parallel for
    for (int i = 0; i < effct_feat_num; i++) {
        //        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(ptpl_list[i].point);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        //        const PointType &norm_p = corr_normvect->points[i];
        //        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        V3D norm_vec(ptpl_list[i].normal);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en) {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        } else {
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        //        ekfom_data.h(i) = -norm_p.intensity;
        float pd2 = norm_vec.x() * ptpl_list[i].point_world.x() + norm_vec.y() * ptpl_list[i].point_world.y() + norm_vec.z() * ptpl_list[i].point_world.z() + ptpl_list[i].d;
        ekfom_data.h(i) = -pd2;

        /*** Covariance ***/
        //        // norm_p中存了匹配的平面法向 还有点面距离
        //        V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
        //        // /*** get the normal vector of closest surface/corner ***/
        //        Eigen::Matrix<double, 1, 6> J_nq;
        //        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        //        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        //        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();
        //
        //        M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
        //        M3D R_cov_Rt = s.rot * cov_lidar * s.rot.conjugate();
        //        // HACK 1. 因为是标量 所以求逆直接用1除
        //        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
        //        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // norm_p中存了匹配的平面法向 还有点面距离
        // V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
        V3D point_world = ptpl_list[i].point_world;
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        // M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
        M3D cov_lidar = ptpl_list[i].cov_lidar;
        M3D R_cov_Rt = s.rot * s.offset_R_L_I * cov_lidar * s.offset_R_L_I.conjugate() * s.rot.conjugate();
        // HACK 1. 因为是标量 所以求逆直接用1除
        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // 计算测量方差R并赋值 目前暂时使用固定值
        // ekfom_data.R(i) = 1.0 / LASER_POINT_COV;
        ekfom_data.R(i) = R_inv;
    }

    // std::printf("Effective Points: %d\n", effct_feat_num);
    res_mean_last = total_residual / effct_feat_num;
    // std::printf("res_mean: %f\n", res_mean_last);
    // std::printf("ef_num: %d\n", effct_feat_num);
}

bool set_kill_on_finish(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res) {
    if (req.data) {
        kill_on_finish = true;
        res.success = true;
        res.message = "kill_on_finish is now set!";
    } else {
        res.success = false;
        res.message = "kill_on_finish not changed";
    }

    return true;
}

bool save_color_cloud_svc_cbk(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
    try {
        colored_map::save_map(save_base_path, true, false);
    } catch (...) {
        res.success = false;
    }

    res.success = true;
    res.message = "saved color pcd in " + save_base_path;

    return true;
}

bool save_lio_cloud_svc_cbk(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
    try {
        string file_name = save_base_path + "/scans.pcd";
        string all_points_dir(file_name);
        pcl::PCDWriter pcd_writer;
        LOG_S(INFO) << "current lio only scan saved to " << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);

        res.message = "saving lio pcd to " + file_name;
    } catch (...) {
        res.success = false;
    }
    res.success = true;

    return true;
}

int main(int argc, char **argv) {
    loguru::init(argc, argv);
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<double>("time_offset", lidar_time_offset, 0.0);

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_diff_lidar_to_imu", time_diff_lidar_to_imu, 0.0);

    // camera params
    nh.param<std::string>("camera/image_topic", cam_topic, "/uncompressed_1");

    // load camera data
    calibration_data::camera.init(nh);

    // Save Params
    nh.param<std::string>("save/base_path", save_base_path, root_dir);
    nh.param<bool>("save/hba_pose_pcd", save_hba_pose_en, false);
    nh.param<bool>("save/image_and_pose", save_image_and_pose_en, false);
    nh.param<bool>("save/openmvs_file", save_openmvs_file_en, false);
    nh.param<bool>("save/cloud_pcd", save_cloud_pcd_en, true);
    nh.param<bool>("save/cloud_ply", save_cloud_ply_en, false);
    nh.param<bool>("save/cloud_lio_pcd", save_cloud_lio_pcd, false);

    nh.param<bool>("common/debug", common::debug_en, false);

    // mapping algorithm params
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<int>("mapping/max_points_size", max_points_size, 100);
    nh.param<int>("mapping/max_cov_points_size", max_cov_points_size, 100);
    nh.param<vector<double>>("mapping/layer_point_size", layer_point_size, vector<double>());
    nh.param<int>("mapping/max_layer", max_layer, 2);
    nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
    nh.param<double>("mapping/down_sample_size", filter_size_surf_min, 0.5);
    std::cout << "filter_size_surf_min:" << filter_size_surf_min << std::endl;
    nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    // noise model params
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    nh.param<double>("noise_model/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("noise_model/acc_cov", acc_cov, 0.1);
    nh.param<double>("noise_model/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("noise_model/b_acc_cov", b_acc_cov, 0.0001);

    // visualization params
    nh.param<bool>("publish/pub_voxel_map", publish_voxel_map, false);
    nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);
    nh.param<bool>("publish/pub_color_map", publish_color_map, false);

    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);
    nh.param<bool>("preprocess/feature_extract_enable", p_pre->feature_enabled, false);
    LOG_S(INFO) << "p_pre->lidar_type " << p_pre->lidar_type << endl;
    for (int i = 0; i < layer_point_size.size(); i++) {
        layer_size.push_back(layer_point_size[i]);
    }

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    _featsArray.reset(new PointCloudXYZINormal());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    // XXX 暂时现在lidar callback中固定转换到IMU系下
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, observation_model_share, NUM_MAX_ITERATIONS, epsi);

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber cam_sub = it.subscribe(cam_topic, 100, &rgb_cam_cbk);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubExtrinsic = nh.advertise<nav_msgs::Odometry>("/Extrinsic", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    ros::Publisher voxel_map_pub =
        nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);

    colored_map::m_pub_rgb_render_pointcloud_ptr_vec.resize(1e3);

    // coloring service
    ros::Publisher pubLaserCloudColor = nh.advertise<sensor_msgs::PointCloud2>("/cloud_colour", 100000);
    if (publish_color_map) {
        std::thread pub_rgb_task(color_service::service_pub_rgb_maps, nh);
        pub_rgb_task.detach();
    }

    // ros service to set exit when buffer is finished
    ros::ServiceServer kill_when_done_svc = nh.advertiseService("set_kill_on_finish", set_kill_on_finish);

    ros::ServiceServer save_lio_cloud_svc = nh.advertiseService("save_lio_cloud", save_lio_cloud_svc_cbk);
    ros::ServiceServer save_color_cloud_svc = nh.advertiseService("save_color_cloud", save_color_cloud_svc_cbk);
    //------------------------------------------------------------------------------------------------------

    std::ofstream pose_file;

    if (save_hba_pose_en) {
        filesystem::remove_all(save_base_path + "/hba_dump/");
        filesystem::create_directories(save_base_path + "/hba_dump/pcd");

        pose_file.open(save_base_path + "/hba_dump/" + "pose.json", std::ios::out);
    }

    if (save_image_and_pose_en) {
        filesystem::remove_all(save_base_path + "/image_pose/");
        filesystem::create_directories(save_base_path + "/image_pose/");
    }

    // for Plane Map
    bool init_map = false;

    double sum_optimize_time = 0, sum_update_time = 0;
    int scan_index = 0;

    // possibly make it work with li init
    // {
    // state_point.offset_R_L_I = Init_LI->get_R_LI();
    // state_point.offset_T_L_I = Init_LI->get_T_LI();
    // state_point.pos = -state_point.rot * state_point.offset_R_L_I.transpose() * state_point.offset_T_L_I +
    //                 state_point.pos; //Body frame is IMU frame in FAST-LIO mode
    // state_point.rot = state_point.rot * state_point.offset_R_L_I.transpose();
    // state_point.grav = Init_LI->get_Grav_L0();
    // state_point.bg = Init_LI->get_gyro_bias();
    // state_point.ba = Init_LI->get_acc_bias();
    // }

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status) {
        if (flg_exit) break;
        ros::spinOnce();

        auto [packages_synced, queue_empty] = sync_packages(Measures);

        if (common::debug_en) {
            LOG_S(INFO) << "packages synced: " << packages_synced << ", queue_empty: " << queue_empty << std::endl;
        }

        if (queue_empty and kill_on_finish) {
            LOG_S(WARNING) << "kill_on_finish is set and queue is empty, gracefully exiting ..." << std::endl;
            break;
        }

        if (packages_synced) {
            if (flg_first_scan) {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL)) {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            // ===============================================================================================================
            // 第一帧 如果ekf初始化了 就初始化voxel地图
            if (flg_EKF_inited && !init_map) {
                PointCloudXYZINormal::Ptr world_lidar(new PointCloudXYZINormal);
                transformLidar(state_point, feats_undistort, world_lidar);
                std::vector<pointWithCov> pv_list;

                std::cout << kf.get_P() << std::endl;
                // 计算第一帧所有点的covariance 并用于构建初始地图
                for (size_t i = 0; i < world_lidar->size(); i++) {
                    pointWithCov pv;
                    pv.point << world_lidar->points[i].x, world_lidar->points[i].y,
                        world_lidar->points[i].z;
                    V3D point_this(feats_undistort->points[i].x,
                                   feats_undistort->points[i].y,
                                   feats_undistort->points[i].z);
                    // if z=0, error will occur in calcBodyCov. To be solved
                    if (point_this[2] == 0) {
                        point_this[2] = 0.001;
                    }
                    M3D cov_lidar = calcBodyCov(point_this, ranging_cov, angle_cov);
                    // 转换到world系
                    M3D cov_world = transformLiDARCovToWorld(point_this, kf, cov_lidar);

                    pv.cov = cov_world;
                    pv_list.push_back(pv);
                    Eigen::Vector3d sigma_pv = pv.cov.diagonal();
                    sigma_pv[0] = sqrt(sigma_pv[0]);
                    sigma_pv[1] = sqrt(sigma_pv[1]);
                    sigma_pv[2] = sqrt(sigma_pv[2]);
                }

                buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                              max_points_size, max_points_size, min_eigen_value,
                              voxel_map);
                LOG_S(INFO) << "build voxel map" << std::endl;

                if (publish_voxel_map) {
                    pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
                    publish_frame_world(pubLaserCloudFull);
                    // publish_frame_body(pubLaserCloudFull_body);
                }
                init_map = true;
                continue;
            }

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);

            feats_down_size = feats_down_body->points.size();
            // 由于点云的body var是一直不变的 因此提前计算 在迭代时可以复用
            var_down_body.clear();
            for (auto &pt : feats_down_body->points) {
                V3D point_this(pt.x, pt.y, pt.z);
                var_down_body.push_back(calcBodyCov(point_this, ranging_cov, angle_cov));
            }

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5) {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            // ===============================================================================================================
            // 开始迭代滤波
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_diagonal();
            //            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            double t_update_end = omp_get_wtime();
            sum_optimize_time += t_update_end - t_update_start;

            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];
            publish_odometry(pubOdomAftMapped);
            //
            if (common::debug_en) {
                std::printf("BA: %.4f %.4f %.4f   BG: %.4f %.4f %.4f   g: %.4f %.4f %.4f\n",
                            kf.get_x().ba.x(), kf.get_x().ba.y(), kf.get_x().ba.z(),
                            kf.get_x().bg.x(), kf.get_x().bg.y(), kf.get_x().bg.z(),
                            kf.get_x().grav.get_vect().x(), kf.get_x().grav.get_vect().y(), kf.get_x().grav.get_vect().z());
            }

            // ===============================================================================================================
            // 更新地图
            /*** add the points to the voxel map ***/
            // 用最新的状态估计将点及点的covariance转换到world系
            std::vector<pointWithCov> pv_list;
            PointCloudXYZINormal::Ptr world_lidar(new PointCloudXYZINormal);
            transformLidar(state_point, feats_down_body, world_lidar);
            for (size_t i = 0; i < feats_down_body->size(); i++) {
                // 保存body系和world系坐标
                pointWithCov pv;
                pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
                // 计算lidar点的cov
                // FIXME 这里错误的使用世界系的点来calcBodyCov时 反倒在某些seq（比如hilti2022的03 15）上效果更好 需要考虑是不是init_plane时使用更大的cov更好
                // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
                // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
                M3D cov_lidar = var_down_body[i];
                // 将body系的var转换到world系
                M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);

                // 最终updateVoxelMap需要用的是world系的point
                pv.cov = cov_world;
                pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
                pv_list.push_back(pv);
            }

            t_update_start = omp_get_wtime();
            std::sort(pv_list.begin(), pv_list.end(), var_contrast);
            updateVoxelMapOMP(pv_list, max_voxel_size, max_layer, layer_size,
                              max_points_size, max_points_size, min_eigen_value,
                              voxel_map);
            t_update_end = omp_get_wtime();
            sum_update_time += t_update_end - t_update_start;

            scan_index++;
            if (common::debug_en) {
                std::printf("Mean  Topt: %.5fs   Tu: %.5fs\n", sum_optimize_time / scan_index, sum_update_time / scan_index);
            }
            // ===============================================================================================================
            // 可视化相关的shit
            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            //
            //            /*** add the feature points to map kdtree ***/
            //            map_incremental();
            //

            color_service::color_point_cloud(
                state_point,
                pubLaserCloudColor,
                Measures,
                feats_undistort,
                first_lidar_time);

            if (save_openmvs_file_en) {
                openmvs_utils::save_state(
                    state_point,
                    feats_undistort);
            }

            if (save_hba_pose_en) {
                pose_file << state_point.pos.x() << " " << state_point.pos.y() << " " << state_point.pos.z() << " " << state_point.rot.w() << " " << state_point.rot.x() << " " << state_point.rot.y() << " " << state_point.rot.z() << std::endl;
                std::string filename = save_base_path + "/hba_dump/" + "pcd/" + to_string(count_) + ".pcd";
                pcl::io::savePCDFileBinary(filename, *feats_undistort);
                count_++;
            }

            // save_image_and_pose(camera_id, src_img, "/home/inkers/adit/fastlio_one_camera_mvs_added/debug/" + std::to_string(global_index));
            if (save_image_and_pose_en) {
                if (Measures.cam.size() > 0) {
                    sensor_msgs::ImageConstPtr latest_image_ptr = Measures.cam.back();
                    cv::Mat latest_img = cv_bridge::toCvShare(latest_image_ptr, "bgr8")->image;
                    colored_map::save_image_and_pose(
                        state_point,
                        latest_img,
                        save_base_path + "/image_pose/" + std::to_string(colored_map::colored_frame_index));
                }
            }

            /******* Publish points *******/
            if (path_en) publish_path(pubPath);
            if (scan_pub_en || save_cloud_lio_pcd) publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            if (publish_voxel_map) {
                pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
            }
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);
        }

        status = ros::ok();
        rate.sleep();
    }

    // close hba file
    if (pose_file.is_open()) {
        pose_file.close();
    }

    if (save_openmvs_file_en) {
        openmvs_map::save_dump(save_base_path);
    }
    if (save_cloud_pcd_en || save_cloud_ply_en) {
        colored_map::save_map(save_base_path, save_cloud_pcd_en, save_cloud_ply_en);
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && save_cloud_lio_pcd) {
        string file_name = save_base_path + "/scans.pcd";
        string all_points_dir(file_name);
        pcl::PCDWriter pcd_writer;
        LOG_S(INFO) << "current lio only scan saved to " << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    return EXIT_SUCCESS;
}
