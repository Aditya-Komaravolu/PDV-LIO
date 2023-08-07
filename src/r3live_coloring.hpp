#pragma once
#include <pcl/io/ply_io.h>

#include <Eigen/Core>
#include <unordered_map>

#include "calibration_data.hpp"
#include "common_lib.h"
#include "loguru.hpp"

namespace coloring_config {
const double image_obs_cov = 15;
const double process_noise_sigma = 0.1;
const double used_fov_margin = 0.005;
int m_pub_pt_minimum_views = 3;
}  // namespace coloring_config

template <typename T>
inline T getSubPixel(const cv::Mat &mat, const double &row, const double &col, double pyramid_layer = 0) {
    int floor_row = floor(row);
    int floor_col = floor(col);
    double frac_row = row - floor_row;
    double frac_col = col - floor_col;
    int ceil_row = floor_row + 1;
    int ceil_col = floor_col + 1;
    if (pyramid_layer != 0) {
        int pos_bias = pow(2, pyramid_layer - 1);
        floor_row -= pos_bias;
        floor_col -= pos_bias;
        ceil_row += pos_bias;
        ceil_row += pos_bias;
    }
    return ((1.0 - frac_row) * (1.0 - frac_col) * (T)mat.ptr<T>(floor_row)[floor_col]) +
           (frac_row * (1.0 - frac_col) * (T)mat.ptr<T>(ceil_row)[floor_col]) +
           ((1.0 - frac_row) * frac_col * (T)mat.ptr<T>(floor_row)[ceil_col]) +
           (frac_row * frac_col * (T)mat.ptr<T>(ceil_row)[ceil_col]);
}

cv::Mat rectifyImage(const cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs) {
    int img_width = image.cols;
    int img_height = image.rows;
    cv::Mat map1, map2;

    cv::initUndistortRectifyMap(camera_matrix, distortion_coeffs, cv::Mat(), camera_matrix, cv::Size(img_width, img_height), CV_16SC2, map1, map2);

    cv::Mat img_pose;
    cv::remap(image, img_pose, map1, map2, cv::INTER_LINEAR);

    return img_pose;
}

inline void image_equalize(cv::Mat &img, int amp) {
    cv::Mat img_temp;
    cv::Size eqa_img_size = cv::Size(std::max(img.cols * 32.0 / 640, 4.0), std::max(img.cols * 32.0 / 640, 4.0));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(amp, eqa_img_size);
    // Equalize gray image.
    clahe->apply(img, img_temp);
    img = img_temp;
}

inline cv::Mat equalize_color_image_Ycrcb(cv::Mat &image) {
    cv::Mat hist_equalized_image;
    cv::cvtColor(image, hist_equalized_image, cv::COLOR_BGR2YCrCb);
    // Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
    std::vector<cv::Mat> vec_channels;
    cv::split(hist_equalized_image, vec_channels);
    // Equalize the histogram of only the Y channel
    //  cv::equalizeHist(vec_channels[0], vec_channels[0]);
    image_equalize(vec_channels[0], 1);
    cv::merge(vec_channels, hist_equalized_image);
    cv::cvtColor(hist_equalized_image, hist_equalized_image, cv::COLOR_YCrCb2BGR);
    return hist_equalized_image;
}

template <typename data_type = float, typename T = void *>
struct hashmap_3d {
    using hash_3d_T = std::unordered_map<data_type, std::unordered_map<data_type, std::unordered_map<data_type, T>>>;
    hash_3d_T m_map_3d_hash_map;
    void insert(const data_type &x, const data_type &y, const data_type &z, const T &target) {
        m_map_3d_hash_map[x][y][z] = target;
    }

    int if_exist(const data_type &x, const data_type &y, const data_type &z) {
        if (m_map_3d_hash_map.find(x) == m_map_3d_hash_map.end()) {
            return 0;
        } else if (m_map_3d_hash_map[x].find(y) == m_map_3d_hash_map[x].end()) {
            return 0;
        } else if (m_map_3d_hash_map[x][y].find(z) == m_map_3d_hash_map[x][y].end()) {
            return 0;
        }
        return 1;
    }

    void clear() {
        m_map_3d_hash_map.clear();
    }

    int total_size() {
        int count = 0;
        for (auto it : m_map_3d_hash_map) {
            for (auto it_it : it.second) {
                for (auto it_it_it : it_it.second) {
                    count++;
                }
            }
        }
        return count;
    }
};

class rgb_pts {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double m_pos[3] = {0};
    double m_rgb[3] = {0};
    double m_cov_rgb[3] = {0};
    double m_gray = 0;
    double m_cov_gray = 0;
    int m_N_gray = 0;
    int m_N_rgb = 0;
    int m_pt_index = 0;
    vec_2 m_img_vel;
    vec_2 m_img_pt_in_last_frame;
    vec_2 m_img_pt_in_current_frame;
    int m_is_out_lier_count = 0;
    cv::Scalar m_dbg_color;
    double m_obs_dis = 0;
    double m_last_obs_time = 0;
    void clear() {
        m_rgb[0] = 0;
        m_rgb[1] = 0;
        m_rgb[2] = 0;
        m_gray = 0;
        m_cov_gray = 0;
        m_N_gray = 0;
        m_N_rgb = 0;
        m_obs_dis = 0;
        m_last_obs_time = 0;
        int r = g_rng.uniform(0, 256);
        int g = g_rng.uniform(0, 256);
        int b = g_rng.uniform(0, 256);
        m_dbg_color = cv::Scalar(r, g, b);
        // m_rgb = vec_3(255, 255, 255);
    };

    rgb_pts() {
        // m_pt_index = g_pts_index++;
        clear();
    };
    ~rgb_pts(){};

    void set_pos(const vec_3 &pos) {
        m_pos[0] = pos(0);
        m_pos[1] = pos(1);
        m_pos[2] = pos(2);
    }
    vec_3 get_pos() {
        return vec_3(m_pos[0], m_pos[1], m_pos[2]);
    }
    vec_3 get_rgb() {
        return vec_3(m_rgb[0], m_rgb[1], m_rgb[2]);
    }
    mat_3_3 get_rgb_cov() {
        mat_3_3 cov_mat = mat_3_3::Zero();
        for (int i = 0; i < 3; i++) {
            cov_mat(i, i) = m_cov_rgb[i];
        }
        return cov_mat;
    }
    pcl::PointXYZI get_pt() {
        pcl::PointXYZI pt;
        pt.x = m_pos[0];
        pt.y = m_pos[1];
        pt.z = m_pos[2];
        return pt;
    }
    int update_rgb(const vec_3 &rgb, const double obs_dis, const vec_3 obs_sigma, const double obs_time) {
        if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2)) {
            return 0;
        }

        if (m_N_rgb == 0) {
            // For first time of observation.
            m_last_obs_time = obs_time;
            m_obs_dis = obs_dis;
            for (int i = 0; i < 3; i++) {
                m_rgb[i] = rgb[i];
                m_cov_rgb[i] = obs_sigma(i);
            }
            m_N_rgb = 1;
            return 0;
        }
        // State estimation for robotics, section 2.2.6, page 37-38
        for (int i = 0; i < 3; i++) {
            m_cov_rgb[i] = (m_cov_rgb[i] + coloring_config::process_noise_sigma * (obs_time - m_last_obs_time));  // Add process noise
            double old_sigma = m_cov_rgb[i];
            m_cov_rgb[i] = sqrt(1.0 / (1.0 / m_cov_rgb[i] / m_cov_rgb[i] + 1.0 / obs_sigma(i) / obs_sigma(i)));
            m_rgb[i] = m_cov_rgb[i] * m_cov_rgb[i] * (m_rgb[i] / old_sigma / old_sigma + rgb(i) / obs_sigma(i) / obs_sigma(i));
            m_rgb[i] = rgb[i];
            m_cov_rgb[i] = obs_sigma(i);
        }

        if (obs_dis < m_obs_dis) {
            m_obs_dis = obs_dis;
        }
        m_last_obs_time = obs_time;
        m_N_rgb++;
        return 1;
    }
};

using rgb_pt_ptr = std::shared_ptr<rgb_pts>;

class rgb_voxel {
   public:
    std::vector<rgb_pt_ptr> m_pts_in_grid;
    double m_last_visited_time = 0;
    rgb_voxel() = default;
    ~rgb_voxel() = default;
    void add_pt(rgb_pt_ptr &rgb_pts) { this->m_pts_in_grid.push_back(rgb_pts); }
};

using rgb_voxel_ptr = std::shared_ptr<rgb_voxel>;
using voxel_set_iterator = std::unordered_set<std::shared_ptr<rgb_voxel>>::iterator;

// R3LIVE CONFIG

struct global_map {
    // [start] config
    double m_recent_visited_voxel_activated_time = 0.0;
    int m_append_global_map_point_step = 1;
    int m_number_of_new_visited_voxel = 0;
    int m_pub_pt_minimum_views = 3;
    // [end] config

    hashmap_3d<long, rgb_pt_ptr> m_hashmap_3d_pts;
    hashmap_3d<long, std::shared_ptr<rgb_voxel>> m_hashmap_voxels;
    std::unordered_set<std::shared_ptr<rgb_voxel>> m_voxels_recent_visited;
    std::shared_ptr<std::mutex> m_mutex_m_box_recent_hitted;
    std::vector<rgb_pt_ptr> m_rgb_pts_vec;
    bool m_in_appending_pts = 0;
    double m_minimum_pts_size = 0.02;  // 5cm minimum distance.
    double m_voxel_resolution = 0.1;
    int m_last_updated_frame_idx = -1;

    global_map() {
        this->m_mutex_m_box_recent_hitted = std::make_shared<std::mutex>();
        this->m_rgb_pts_vec.reserve(1e8);
    }

    template <typename T>
    int append_points_to_global_map(
        pcl::PointCloud<T> &pc_in, double added_time, std::vector<std::shared_ptr<rgb_pts>> *pts_added_vec, int step) {
        this->m_in_appending_pts = 1;
        int acc = 0;
        int rej = 0;
        std::unordered_set<std::shared_ptr<rgb_voxel>> voxels_recent_visited;
        if (m_recent_visited_voxel_activated_time == 0) {
            voxels_recent_visited.clear();
        } else {
            m_mutex_m_box_recent_hitted->lock();
            voxels_recent_visited = m_voxels_recent_visited;
            m_mutex_m_box_recent_hitted->unlock();
            for (voxel_set_iterator it = voxels_recent_visited.begin(); it != voxels_recent_visited.end();) {
                if (added_time - (*it)->m_last_visited_time > m_recent_visited_voxel_activated_time) {
                    it = voxels_recent_visited.erase(it);
                    continue;
                }
                it++;
            }
            LOG_S(INFO) << "Restored voxel number = " << voxels_recent_visited.size() << std::endl;
        }
        int number_of_voxels_before_add = voxels_recent_visited.size();
        int pt_size = pc_in.points.size();
        // step = 4;
        for (int pt_idx = 0; pt_idx < pt_size; pt_idx += step) {
            int add = 1;
            int grid_x = std::round(pc_in.points[pt_idx].x / m_minimum_pts_size);
            int grid_y = std::round(pc_in.points[pt_idx].y / m_minimum_pts_size);
            int grid_z = std::round(pc_in.points[pt_idx].z / m_minimum_pts_size);
            int box_x = std::round(pc_in.points[pt_idx].x / m_voxel_resolution);
            int box_y = std::round(pc_in.points[pt_idx].y / m_voxel_resolution);
            int box_z = std::round(pc_in.points[pt_idx].z / m_voxel_resolution);
            if (m_hashmap_3d_pts.if_exist(grid_x, grid_y, grid_z)) {
                add = 0;
                if (pts_added_vec != nullptr) {
                    pts_added_vec->push_back(m_hashmap_3d_pts.m_map_3d_hash_map[grid_x][grid_y][grid_z]);
                }
            }
            rgb_voxel_ptr box_ptr;
            if (!m_hashmap_voxels.if_exist(box_x, box_y, box_z)) {
                std::shared_ptr<rgb_voxel> box_rgb = std::make_shared<rgb_voxel>();
                m_hashmap_voxels.insert(box_x, box_y, box_z, box_rgb);
                box_ptr = box_rgb;
            } else {
                box_ptr = m_hashmap_voxels.m_map_3d_hash_map[box_x][box_y][box_z];
            }
            voxels_recent_visited.insert(box_ptr);
            box_ptr->m_last_visited_time = added_time;
            if (add == 0) {
                rej++;
                continue;
            }
            acc++;
            std::shared_ptr<rgb_pts> pt_rgb = std::make_shared<rgb_pts>();
            pt_rgb->set_pos(vec_3(pc_in.points[pt_idx].x, pc_in.points[pt_idx].y, pc_in.points[pt_idx].z));
            pt_rgb->m_pt_index = m_rgb_pts_vec.size();
            m_rgb_pts_vec.push_back(pt_rgb);
            m_hashmap_3d_pts.insert(grid_x, grid_y, grid_z, pt_rgb);
            box_ptr->add_pt(pt_rgb);
            if (pts_added_vec != nullptr) {
                pts_added_vec->push_back(pt_rgb);
            }
        }
        m_in_appending_pts = 0;
        m_mutex_m_box_recent_hitted->lock();
        m_voxels_recent_visited = voxels_recent_visited;
        m_mutex_m_box_recent_hitted->unlock();
        return (m_voxels_recent_visited.size() - number_of_voxels_before_add);
    }

    void save_to_pcd(std::string dir_name, std::string file_name, int save_pts_with_views, bool save_pcd = true, bool save_ply = true) {
        std::string file_path = std::string(dir_name).append(file_name);

        // std::cout << "save rgb points to " << file_path << ".pcd" << std::endl;

        pcl::PointCloud<pcl::PointXYZRGB> pc_rgb;
        long pt_size = m_rgb_pts_vec.size();
        pc_rgb.resize(pt_size);
        long pt_count = 0;
        for (long i = pt_size - 1; i > 0; i--)
        // for (int i = 0; i  <  pt_size; i++)
        {
            if (i % 1000 == 0) {
                LOG_S(INFO) << "saving rgb cloud map " << (int)((pt_size - 1 - i) * 100.0 / (pt_size - 1)) << " % ...\r";
            }

            if (m_rgb_pts_vec[i]->m_N_rgb < 1) {
                continue;
            }
            pcl::PointXYZRGB pt;
            pc_rgb.points[pt_count].x = m_rgb_pts_vec[i]->m_pos[0];
            pc_rgb.points[pt_count].y = m_rgb_pts_vec[i]->m_pos[1];
            pc_rgb.points[pt_count].z = m_rgb_pts_vec[i]->m_pos[2];
            pc_rgb.points[pt_count].r = m_rgb_pts_vec[i]->m_rgb[2];
            pc_rgb.points[pt_count].g = m_rgb_pts_vec[i]->m_rgb[1];
            pc_rgb.points[pt_count].b = m_rgb_pts_vec[i]->m_rgb[0];
            pt_count++;
        }
        LOG_S(INFO) << "saving offline map 100% ..." << std::endl;
        pc_rgb.resize(pt_count);
        LOG_S(INFO) << "total have " << pt_count << " points." << std::endl;
        if (save_pcd) {
            LOG_S(INFO) << "saving pcd to: " << file_path << ".pcd" << std::endl;
            pcl::io::savePCDFileBinary(std::string(file_path).append(".pcd"), pc_rgb);
        }

        if (save_ply) {
            LOG_S(INFO) << "saving ply to: " << file_path << ".ply" << std::endl;
            pcl::io::savePLYFileBinary(std::string(file_path).append(".ply"), pc_rgb);
        }
    }
};

namespace colored_map {
global_map m_map_rgb_pts;
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullResColor = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
std::vector<rgb_voxel_ptr> g_voxel_for_render;
std::atomic<long> render_pts_count;
std::vector<std::shared_ptr<ros::Publisher>> m_pub_rgb_render_pointcloud_ptr_vec;
int colored_frame_index = 0;

void save_map(std::string save_dir, bool save_pcd = true, bool save_ply = false) {
    std::cout << "saving coloured point cloud" << std::endl;
    m_map_rgb_pts.save_to_pcd(save_dir, std::string("/rgb_pt"), coloring_config::m_pub_pt_minimum_views);
}

void save_image_and_pose(const state_ikfom state_point, cv::Mat image_to_save, std::string save_path) {
    auto rot_mat = state_point.rot.toRotationMatrix();
    auto trans_vec = state_point.pos;

    Eigen::Quaterniond q;

    auto rot_wrt_cam = rot_mat * calibration_data::camera.extrinsicMat_R_eigen;
    auto trans_wrt_cam = rot_mat * calibration_data::camera.extrinsicMat_T_eigen + trans_vec;

    // Convert 3x3 Eigen matrix to quaternion
    // std::cout<<"Rotation matrix "<<rot_wrt_cam<<std::endl;
    q = Eigen::Quaterniond(rot_wrt_cam);

    // Save quaternion and translation to a txt file based on the camera frame index
    std::string txt_file_name = std::string(save_path).append(".txt");
    std::string image_file_name = std::string(save_path).append(".png");
    FILE *fp = fopen(txt_file_name.c_str(), "w+");
    if (fp) {
        fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf\r\n", q.w(), q.x(), q.y(), q.z(),
                trans_wrt_cam(0), trans_wrt_cam(1), trans_wrt_cam(2));
        fclose(fp);
    }
    // print the mats

    // std::cout<<"cam 1 extrinsic mat "<<cam_1_extrinsicMat_RT<<std::endl;
    // std::cout<<"cam 1 instrinsics mat"<<cam_1_intrisicMat<<std::endl;

    auto rectified_image = rectifyImage(image_to_save, calibration_data::camera.intrisicMat, calibration_data::camera.distCoeffs);
    cv::imwrite(image_file_name, rectified_image);
}

}  // namespace colored_map
