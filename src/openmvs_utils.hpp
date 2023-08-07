#pragma once

#include <common_lib.h>
#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/Vector3.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <so3_math.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Eigen>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common_lib.h"
#include "use-ikfom.hpp"

using namespace std;

// For MVS dump : start
class XYZPointDump {
   public:
    uint32_t m_point_idx;
    std::array<double, 3> m_pos = {};

    XYZPointDump(uint32_t point_idx, double x, double y, double z) {
        m_pos[0] = x;
        m_pos[1] = y;
        m_pos[2] = z;
        m_point_idx = point_idx;
    }
    friend ostream &operator<<(ostream &os, const XYZPointDump &xyz);

   private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar &m_point_idx;
        ar &m_pos;
    }
};

ostream &operator<<(ostream &os, const XYZPointDump &xyz) {
    os << "Point [" << xyz.m_pos[0] << ", " << xyz.m_pos[1] << ", " << xyz.m_pos[2] << "]";
    return os;
}

class ViewFrameDump {
   public:
    uint32_t m_frame_idx;
    std::array<double, 3> m_pos = {};
    std::array<double, 4> m_quat = {};

    ViewFrameDump(uint32_t frame_idx, double x, double y, double z, double qw, double qx, double qy, double qz) {
        m_frame_idx = frame_idx;

        m_pos[0] = x;
        m_pos[1] = y;
        m_pos[2] = z;

        m_quat[0] = qw;
        m_quat[1] = qx;
        m_quat[2] = qy;
        m_quat[3] = qz;
    }

    friend ostream &operator<<(ostream &os, const ViewFrameDump &vfd);

   private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar &m_frame_idx;
        ar &m_pos;
        ar &m_quat;
    }
};

ostream &operator<<(ostream &os, const ViewFrameDump &vfd) {
    os << "ViewFrame pos: [" << vfd.m_pos[0] << ", " << vfd.m_pos[1] << ", " << vfd.m_pos[2] << "]"
       << " quat: [" << vfd.m_quat[0] << ", " << vfd.m_quat[1] << ", " << vfd.m_quat[2] << ", " << vfd.m_quat[3] << "]";
    return os;
}

class PointCloudXYZDump {
   public:
    uint32_t num_points = 0;
    uint32_t num_views = 0;
    // vector of points
    std::vector<XYZPointDump> m_points;
    // vector of views
    std::vector<ViewFrameDump> m_pose_vec;
    // point to view, given point, gives the list of views
    std::unordered_map<uint32_t, std::vector<uint32_t>> m_pts_with_view;
    // given a view, gives vector of points (indices)
    std::vector<std::vector<uint32_t>> m_pts_in_views_vec;

    friend ostream &operator<<(ostream &os, const PointCloudXYZDump dump);

    void dump_to_file(std::string file_name, int if_bin = 1) {
        std::ofstream ofs(file_name);

        if (ofs.is_open()) {
            if (if_bin) {
                boost::archive::binary_oarchive oa(ofs);
                oa << *this;
                ofs.close();
            } else {
                boost::archive::text_oarchive oa(ofs);
                oa << *this;
                ofs.close();
            }
        } else {
            LOG_S(WARNING) << "Dump to file [" << file_name << "] fail!, cannot open file" << endl;
        }
    }

   private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar &num_points;
        ar &num_views;
        ar &m_points;
        ar &m_pose_vec;
        ar &m_pts_with_view;
        ar &m_pts_in_views_vec;
    }
};

ostream &operator<<(ostream &os, const PointCloudXYZDump dump) {
    os << "Map Dump: "
       << "\n"
       << "m_points: "
       << "\n";

    for (auto &point : dump.m_points) {
        os << point << "\n";
    }

    os << "m_pose_vec"
       << "\n";
    for (auto &view : dump.m_pose_vec) {
        os << view << "\n";
    }

    return os;
}

namespace openmvs_map {
PointCloudXYZDump point_cloud_dump;
void save_dump(std::string dump_dir) {
    // std::cout << point_cloud_dump << "\n";
    point_cloud_dump.num_points = point_cloud_dump.m_points.size();
    point_cloud_dump.num_views = point_cloud_dump.m_pose_vec.size();
    // std::string file_name = "/home/inkers/adit/catkin_pv_lio/test.pv_lio";
    std::string file_name = dump_dir + "test.pv_lio";

    point_cloud_dump.dump_to_file(file_name);
    LOG_S(INFO) << "saved openmvs dump to " << file_name << "\n";
}
}  // namespace openmvs_map

void RGBpointBodyToWorld(const state_ikfom state_point, PointType const *const pi, PointType *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->rgba = pi->rgba;
}

void RGBpointBodyToWorld(const state_ikfom &state_point, pcl::PointXYZINormal const *const pi, pcl::PointXYZINormal *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

namespace openmvs_utils {
void save_state(const state_ikfom state_point, PointCloudXYZINormal::Ptr feats_undistort) {
    vect3 pos_lid;
    geometry_msgs::Quaternion geoQuat;

    SO3 flio_to_r3live(0.5, -0.5, 0.5, -0.5);
    auto new_rot = flio_to_r3live * state_point.rot;
    // new_rot = new_rot.inverse();
    new_rot = new_rot;
    pos_lid = state_point.pos + new_rot * state_point.offset_T_L_I;
    pos_lid = -(new_rot.inverse() * state_point.pos);
    auto inv_rot = new_rot;
    geoQuat.x = inv_rot.coeffs()[0];
    geoQuat.y = inv_rot.coeffs()[1];
    geoQuat.z = inv_rot.coeffs()[2];
    geoQuat.w = inv_rot.coeffs()[3];

    // save position: state_point.pos
    // save quaternion: state_point.rot
    // current lidar frame get from publish_frame_world

    {
        // std::cout << "pos: [" << pos_lid[0] << ", " << pos_lid[1] << ", " << pos_lid[2] << "]" << "\n";
        // std::cout << "quat: [" << geoQuat.w << ", " << geoQuat.x << ", " << geoQuat.y << ", " << geoQuat.z << "]" << "\n";
        // std::cout << std::endl;

        auto current_view_idx = openmvs_map::point_cloud_dump.m_pose_vec.size();

        std::shared_ptr<ViewFrameDump> view_frame = std::make_shared<ViewFrameDump>(
            current_view_idx,
            pos_lid[0],
            pos_lid[1],
            pos_lid[2],
            geoQuat.w,
            geoQuat.x,
            geoQuat.y,
            geoQuat.z);

        int size = feats_undistort->points.size();
        PointCloudXYZINormal::Ptr laserCloudWorld(new PointCloudXYZINormal(size, 1));

        std::vector<uint32_t> points_in_view;

        for (int i = 0; i < size; i++) {
            RGBpointBodyToWorld(state_point, &feats_undistort->points[i], &laserCloudWorld->points[i]);
            // laserCloudWorld->points[i].x = feats_undistort->points[i].x;
            // laserCloudWorld->points[i].y = feats_undistort->points[i].y;
            // laserCloudWorld->points[i].z = feats_undistort->points[i].z;
            // laserCloudWorld->points[i].intensity = feats_undistort->points[i].intensity;
            // laserCloudWorld->points[i].rgba = feats_undistort->points[i].rgba;

            auto current_point_idx = openmvs_map::point_cloud_dump.m_points.size();
            std::shared_ptr<XYZPointDump> xyz_point_dump = std::make_shared<XYZPointDump>(current_point_idx, laserCloudWorld->points[i].x, laserCloudWorld->points[i].y, laserCloudWorld->points[i].z);
            openmvs_map::point_cloud_dump.m_points.push_back(*xyz_point_dump);

            openmvs_map::point_cloud_dump.m_pts_with_view[current_point_idx] = {(unsigned int)current_view_idx};

            points_in_view.push_back(current_point_idx);
        }

        openmvs_map::point_cloud_dump.m_pts_in_views_vec.push_back(points_in_view);

        openmvs_map::point_cloud_dump.m_pose_vec.push_back(*view_frame);
    }
}
}  // namespace openmvs_utils