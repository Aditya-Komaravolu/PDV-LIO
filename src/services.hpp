#pragma once

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <thread>

#include "calibration_data.hpp"
#include "common_lib.h"
#include "custom_point.hpp"
#include "openmvs_utils.hpp"
#include "r3live_coloring.hpp"
#include "use-ikfom.hpp"

// #include "IMU_Processing.hpp"

namespace color_service {

void service_pub_rgb_maps(ros::NodeHandle nh) {
    int last_publish_map_idx = -3e8;
    int sleep_time_aft_pub = 10;
    int number_of_pts_per_topic = 1000;
    if (number_of_pts_per_topic < 0) {
        return;
    }

    while (true) {
        ros::spinOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        pcl::PointCloud<pcl::PointXYZRGB> pc_rgb;
        sensor_msgs::PointCloud2 ros_pc_msg;
        int pts_size = colored_map::m_map_rgb_pts.m_rgb_pts_vec.size();
        pc_rgb.resize(number_of_pts_per_topic);
        // for (int i = pts_size - 1; i > 0; i--)
        int pub_idx_size = 0;
        int cur_topic_idx = 0;
        if (last_publish_map_idx == colored_map::m_map_rgb_pts.m_last_updated_frame_idx) {
            continue;
        }
        last_publish_map_idx = colored_map::m_map_rgb_pts.m_last_updated_frame_idx;
        for (int i = 0; i < pts_size; i++) {
            if (colored_map::m_map_rgb_pts.m_rgb_pts_vec[i]->m_N_rgb < 1) {
                continue;
            }
            pc_rgb.points[pub_idx_size].x = colored_map::m_map_rgb_pts.m_rgb_pts_vec[i]->m_pos[0];
            pc_rgb.points[pub_idx_size].y = colored_map::m_map_rgb_pts.m_rgb_pts_vec[i]->m_pos[1];
            pc_rgb.points[pub_idx_size].z = colored_map::m_map_rgb_pts.m_rgb_pts_vec[i]->m_pos[2];
            pc_rgb.points[pub_idx_size].r = colored_map::m_map_rgb_pts.m_rgb_pts_vec[i]->m_rgb[2];
            pc_rgb.points[pub_idx_size].g = colored_map::m_map_rgb_pts.m_rgb_pts_vec[i]->m_rgb[1];
            pc_rgb.points[pub_idx_size].b = colored_map::m_map_rgb_pts.m_rgb_pts_vec[i]->m_rgb[0];
            // pc_rgb.points[ pub_idx_size ].r = 255;
            // pc_rgb.points[ pub_idx_size ].g = 255;
            // pc_rgb.points[ pub_idx_size ].b = 255;
            // pc_rgb.points[i].intensity = m_map_rgb_pts.m_rgb_pts_vec[i]->m_obs_dis;
            pub_idx_size++;
            if (pub_idx_size == number_of_pts_per_topic) {
                pub_idx_size = 0;
                pcl::toROSMsg(pc_rgb, ros_pc_msg);
                ros_pc_msg.header.frame_id = "camera_init";
                ros_pc_msg.header.stamp = ros::Time::now();
                if (colored_map::m_pub_rgb_render_pointcloud_ptr_vec[cur_topic_idx] == nullptr) {
                    colored_map::m_pub_rgb_render_pointcloud_ptr_vec[cur_topic_idx] =
                        std::make_shared<ros::Publisher>(nh.advertise<sensor_msgs::PointCloud2>(
                            std::string("/RGB_map_").append(std::to_string(cur_topic_idx)), 100));
                }
                colored_map::m_pub_rgb_render_pointcloud_ptr_vec[cur_topic_idx]->publish(ros_pc_msg);

                // std::cout << "service_pub_rgb_maps " << pub_idx_size << " with timestamp " << ros_pc_msg.header.stamp << std::endl;
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_aft_pub));
                ros::spinOnce();
                cur_topic_idx++;
            }
        }

        pc_rgb.resize(pub_idx_size);
        pcl::toROSMsg(pc_rgb, ros_pc_msg);
        ros_pc_msg.header.frame_id = "camera_init";
        ros_pc_msg.header.stamp = ros::Time::now();
        if (colored_map::m_pub_rgb_render_pointcloud_ptr_vec[cur_topic_idx] == nullptr) {
            colored_map::m_pub_rgb_render_pointcloud_ptr_vec[cur_topic_idx] =
                std::make_shared<ros::Publisher>(nh.advertise<sensor_msgs::PointCloud2>(
                    std::string("/RGB_map_").append(std::to_string(cur_topic_idx)), 100));
        }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_aft_pub));
        ros::spinOnce();
        colored_map::m_pub_rgb_render_pointcloud_ptr_vec[cur_topic_idx]->publish(ros_pc_msg);

        // std::cout << "service_pub_rgb_maps with timestamp " << ros_pc_msg.header.stamp << std::endl;

        cur_topic_idx++;
        if (cur_topic_idx >= 45)  // Maximum pointcloud topics = 45.
        {
            number_of_pts_per_topic *= 1.5;
            sleep_time_aft_pub *= 1.5;
        }
    }
}

static inline void thread_render_pts_in_voxel(const calibration_data::camera_t &camera, const state_ikfom &state_point, const int &pt_start, const int &pt_end, const cv::Mat &img_ptr,
                                              const std::vector<rgb_voxel_ptr> *voxels_for_render, const double obs_time) {
    vec_3 pt_w;
    vec_3 rgb_color;
    double u, v;
    double pt_cam_norm;
    double scale = 1.0;

    auto rot_mat = state_point.rot.toRotationMatrix();
    auto trans_vec = state_point.pos;
    auto trans_wrt_cam = rot_mat * camera.extrinsicMat_T_eigen + trans_vec;

    auto cam_1_extrinsicMat_R_eigen_inverted = camera.extrinsicMat_R_eigen.inverse();
    Eigen::Matrix3d m_pose_w2c_q = rot_mat * cam_1_extrinsicMat_R_eigen_inverted;
    auto m_pose_w2c_t = rot_mat * camera.extrinsicMat_T_eigen + trans_vec;

    Eigen::Matrix3d m_pose_c2w_q = m_pose_w2c_q.inverse();
    auto m_pose_c2w_t = -(m_pose_w2c_q.inverse() * m_pose_w2c_t);

    for (int voxel_idx = pt_start; voxel_idx < pt_end; voxel_idx++) {
        // continue;
        rgb_voxel_ptr voxel_ptr = (*voxels_for_render)[voxel_idx];
        for (int pt_idx = 0; pt_idx < voxel_ptr->m_pts_in_grid.size(); pt_idx++) {
            pt_w = voxel_ptr->m_pts_in_grid[pt_idx]->get_pos();
            // if (img_ptr->project_3d_point_in_this_img(pt_w, u, v, nullptr, 1.0) == false) {
            //     continue;
            // }

            vec_3 pt_cam = m_pose_c2w_q * pt_w + m_pose_c2w_t;
            // vec_3 pt_cam = m_pose_w2c_q * pt_w + m_pose_w2c_t;

            // void Image_frame::refresh_pose_for_projection() {
            //     m_pose_c2w_q = m_pose_w2c_q.inverse();
            //     m_pose_c2w_t = -(m_pose_w2c_q.inverse() * m_pose_w2c_t);
            //     m_if_have_set_pose = 1;
            // }

            // void Image_frame::set_pose(const eigen_q &pose_w2c_q, const vec_3 &pose_w2c_t) {
            //     m_pose_w2c_q = pose_w2c_q;
            //     m_pose_w2c_t = pose_w2c_t;
            //     refresh_pose_for_projection();
            // }

            // mat_3_3 rot_mat = state.rot_end;
            // vec_3   t_vec = state.pos_end;
            // vec_3   pose_t = rot_mat * state.pos_ext_i2c + t_vec;
            // mat_3_3 R_w2c = rot_mat * state.rot_ext_i2c;

            // image_pose->set_pose( eigen_q( R_w2c ), pose_t );

            if (pt_cam(2) < 0.001) {
                // std::cout<<"PT CAM ERROR";
                // if the point is behind the camera we don't consider it
                continue;
            }

            // std::cout << "cam_1_fx: " << cam_1_fx << std::endl;
            // std::cout << "cam_1_fy: " << cam_1_fy << std::endl;
            // std::cout << "cam_1_cx: " << cam_1_cx << std::endl;
            // std::cout << "cam_1_cy: " << cam_1_cy << std::endl;

            // std::cout << "pt_cam(0): " << pt_cam(0) << std::endl;
            // std::cout << "pt_cam(1): " << pt_cam(1) << std::endl;

            u = (pt_cam(0) * camera.fx / pt_cam(2) + camera.cx) * scale;
            v = (pt_cam(1) * camera.fy / pt_cam(2) + camera.cy) * scale;

            int m_img_cols = img_ptr.cols;
            int m_img_rows = img_ptr.rows;

            if (((u / scale >= (coloring_config::used_fov_margin * m_img_cols + 1)) && (std::ceil(u / scale) < ((1 - coloring_config::used_fov_margin) * m_img_cols)) &&
                 (v / scale >= (coloring_config::used_fov_margin * m_img_rows + 1)) && (std::ceil(v / scale) < ((1 - coloring_config::used_fov_margin) * m_img_rows)))) {
            } else {
                // std::cout<<"OUT OF FOV";
                continue;
            }

            pt_cam_norm = (pt_w - trans_wrt_cam).norm();
            // double gray = img_ptr->get_grey_color(u, v, 0);
            // pts_for_render[i]->update_gray(gray, pt_cam_norm);
            // printf("pt_cam:: %f\n",pt_cam_nor \nm);
            // printf("obs time :: %f\n",obs_tim \ne);
            // rgb_color = img_ptr->get_rgb(u, v, 0);
            cv::Vec3b rgb = getSubPixel<cv::Vec3b>(img_ptr, v, u, 0);
            rgb_color[0] = (float)rgb[0];
            rgb_color[1] = (float)rgb[1];
            rgb_color[2] = (float)rgb[2];

            if (voxel_ptr->m_pts_in_grid[pt_idx]->update_rgb(rgb_color, pt_cam_norm, vec_3(coloring_config::image_obs_cov, coloring_config::image_obs_cov, coloring_config::image_obs_cov), obs_time)) {
                colored_map::render_pts_count++;
            }
        }
    }
}

void render_pts_in_voxels_mp(
    const calibration_data::camera_t &camera,
    const state_ikfom &state_point,
    cv::Mat &rgb_image,
    std::unordered_set<rgb_voxel_ptr> *_voxels_for_render, const double &obs_time) {
    auto src_img = rgb_image;
    src_img = rectifyImage(src_img, camera.intrisicMat, camera.distCoeffs);
    src_img = equalize_color_image_Ycrcb(src_img);

    colored_map::g_voxel_for_render.clear();
    for (voxel_set_iterator it = (*_voxels_for_render).begin(); it != (*_voxels_for_render).end(); it++) {
        colored_map::g_voxel_for_render.push_back(*it);
    }
    std::vector<std::future<double>> results;
    int numbers_of_voxels = colored_map::g_voxel_for_render.size();
    colored_map::render_pts_count = 0;

    int num_of_threads = std::min(8 * 2, (int)numbers_of_voxels);
    // results.clear();
    results.resize(num_of_threads);

    // #pragma omp parallel for
    std::vector<std::thread> render_threads;
    for (int thr = 0; thr < num_of_threads; thr++) {
        // cv::Range range(thr * pt_size / num_of_threads, (thr + 1) * pt_size / num_of_threads);
        int start = thr * numbers_of_voxels / num_of_threads;
        int end = (thr + 1) * numbers_of_voxels / num_of_threads;
        render_threads.push_back(std::thread(thread_render_pts_in_voxel, camera, state_point, start, end, src_img, &colored_map::g_voxel_for_render, obs_time));
    }

    for (auto &t : render_threads) {
        t.join();
    }

    // std::cout << "render_pts_count: " << render_pts_count << std::endl;
}

void service_render_update(const calibration_data::camera_t &camera, const state_ikfom &state_point, cv::Mat &rgb_image, std::unordered_set<rgb_voxel_ptr> *_voxels_for_render, const double &obs_time) {
    render_pts_in_voxels_mp(camera, state_point, rgb_image, _voxels_for_render, obs_time);
}

void publish_render_pts(ros::Publisher &pts_pub, global_map &m_map_rgb_pts) {
    pcl::PointCloud<pcl::PointXYZRGB> pc_rgb;
    sensor_msgs::PointCloud2 ros_pc_msg;
    pc_rgb.reserve(1e7);
    m_map_rgb_pts.m_mutex_m_box_recent_hitted->lock();
    std::unordered_set<std::shared_ptr<rgb_voxel>> boxes_recent_hitted = m_map_rgb_pts.m_voxels_recent_visited;
    m_map_rgb_pts.m_mutex_m_box_recent_hitted->unlock();

    for (voxel_set_iterator it = boxes_recent_hitted.begin(); it != boxes_recent_hitted.end(); it++) {
        for (int pt_idx = 0; pt_idx < (*it)->m_pts_in_grid.size(); pt_idx++) {
            pcl::PointXYZRGB pt;
            std::shared_ptr<rgb_pts> rgb_pt = (*it)->m_pts_in_grid[pt_idx];
            pt.x = rgb_pt->m_pos[0];
            pt.y = rgb_pt->m_pos[1];
            pt.z = rgb_pt->m_pos[2];
            pt.r = rgb_pt->m_rgb[2];
            pt.g = rgb_pt->m_rgb[1];
            pt.b = rgb_pt->m_rgb[0];
            if (rgb_pt->m_N_rgb > coloring_config::m_pub_pt_minimum_views) {
                pc_rgb.points.push_back(pt);
            }
        }
    }
    pcl::toROSMsg(pc_rgb, ros_pc_msg);
    ros_pc_msg.header.frame_id = "camera_init";  // world; camera_init
    ros_pc_msg.header.stamp = ros::Time::now();  //.fromSec(last_timestamp_lidar);
    // std::cout << "publish_render_pts with timestamp " << ros_pc_msg.header.stamp << std::endl;
    pts_pub.publish(ros_pc_msg);
}

void color_point_cloud(state_ikfom &state_point, ros::Publisher &pubLaserCloudColor, MeasureGroup &Measures, PointCloudXYZINormal::Ptr feats_undistort, double first_lidar_time) {
    // if (std::abs(Measures.time_buffer.back() - Measures.cam.back()->header.stamp.toSec()) > 0.05) {
    //     LOG_S(WARNING) << "lidar and camera too much time offset, skip coloring" << std::endl;
    //     return;
    // }
    PointCloudXYZINormal::Ptr laserTemp(feats_undistort);  // dense is TRUE
    pcl::PointXYZINormal temp_point;
    pcl::PointXYZINormal temp_point_type;
    colored_map::laserCloudFullResColor->clear();
    for (int i = 0; i < laserTemp->size(); i++) {
        RGBpointBodyToWorld(state_point, &laserTemp->points[i], &temp_point_type);
        temp_point.x = temp_point_type.x;
        temp_point.y = temp_point_type.y;
        temp_point.z = temp_point_type.z;
        temp_point.intensity = temp_point_type.intensity;
        colored_map::laserCloudFullResColor->push_back(temp_point);
    }
    colored_map::m_map_rgb_pts.m_number_of_new_visited_voxel = colored_map::m_map_rgb_pts.append_points_to_global_map(
        *colored_map::laserCloudFullResColor, Measures.lidar_end_time - first_lidar_time, nullptr,
        colored_map::m_map_rgb_pts.m_append_global_map_point_step);
    if (Measures.cam.size() > 0) {
        sensor_msgs::ImageConstPtr latest_image_ptr = Measures.cam.back();
        cv::Mat latest_img = cv_bridge::toCvShare(latest_image_ptr, "bgr8")->image;
        const double latest_img_time = latest_image_ptr->header.stamp.toSec();
        // std::cout << "image time: " << latest_img_time << std::endl;
        color_service::service_render_update(
            calibration_data::camera,
            state_point,
            latest_img,
            &colored_map::m_map_rgb_pts.m_voxels_recent_visited,
            latest_img_time);
        colored_map::m_map_rgb_pts.m_last_updated_frame_idx = colored_map::colored_frame_index;
    }
    // color_service::publish_render_pts(
    //     pubLaserCloudColor,
    //     colored_map::m_map_rgb_pts);
}
}  // namespace color_service