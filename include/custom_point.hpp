#pragma once
#include <pcl/register_point_struct.h>

#include <Eigen/Core>

namespace pcl {
struct PointXYZRGBINormal;
}

#include "custom_point_def.hpp"

POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::PointXYZRGBINormal,
                                  (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(float, normal_y, normal_y)(float, normal_z, normal_z)(std::uint32_t, rgba, rgba)(float, intensity, intensity)(float, curvature, curvature));
