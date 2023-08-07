#ifndef COMMON_LIB_H
#define COMMON_LIB_H
#define PCL_NO_PRECOMPILE  // !! BEFORE ANY PCL INCLUDE!!

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <so3_math.h>

#include <Eigen/Eigen>
// #include <fast_lio/Pose6D.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen_conversions/eigen_msg.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <pcl/io/pcd_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <stdio.h>
#include <string.h>
#include <tf/transform_broadcaster.h>

#include <future>
#include <opencv2/opencv.hpp>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "custom_point.hpp"

using namespace std;
using namespace Eigen;

#define USE_IKFOM

cv::RNG g_rng = cv::RNG(0);

#define PI_M (3.14159265358)
#define G_m_s2 (9.81)    // Gravaty const in GuangDong/China
#define DIM_STATE (18)   // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12)  // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN (6.0)
#define LIDAR_SP_LEN (2)
#define INIT_COV (1)
#define NUM_MATCH_POINTS (5)
#define MAX_MEAS_DIM (10000)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define CONSTRAIN(v, min, max) ((v > min) ? ((v < max) ? v : max) : min)
#define ARRAY_FROM_EIGEN(mat) mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat) vector<decltype(mat)::Scalar>(mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))

typedef pcl::PointXYZRGBINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZRGBI;
typedef pcl::PointCloud<pcl::PointXYZINormal> PointCloudXYZINormal;
typedef vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;

#define MD(a, b) Matrix<double, (a), (b)>
#define VD(a) Matrix<double, (a), 1>
#define MF(a, b) Matrix<float, (a), (b)>
#define VF(a) Matrix<float, (a), 1>

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

struct MeasureGroup  // Lidar data and imu dates for the curent process
{
    MeasureGroup() {
        lidar_beg_time = 0.0;
        this->lidar.reset(new PointCloudXYZINormal());
    };
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudXYZINormal::Ptr lidar;
    deque<sensor_msgs::Imu::ConstPtr> imu;
    deque<sensor_msgs::ImageConstPtr> cam;
};

struct Pose6D {
    double offset_time;
    double acc[3], gyr[3], vel[3], pos[3], rot[9];
};

struct StatesGroup {
    StatesGroup() {
        this->rot_end = M3D::Identity();
        this->pos_end = Zero3d;
        this->vel_end = Zero3d;
        this->bias_g = Zero3d;
        this->bias_a = Zero3d;
        this->gravity = Zero3d;
        this->cov = MD(DIM_STATE, DIM_STATE)::Identity() * INIT_COV;
        this->cov.block<9, 9>(9, 9) = MD(9, 9)::Identity() * 0.00001;
    };

    StatesGroup(const StatesGroup &b) {
        this->rot_end = b.rot_end;
        this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g = b.bias_g;
        this->bias_a = b.bias_a;
        this->gravity = b.gravity;
        this->cov = b.cov;
    };

    StatesGroup &operator=(const StatesGroup &b) {
        this->rot_end = b.rot_end;
        this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g = b.bias_g;
        this->bias_a = b.bias_a;
        this->gravity = b.gravity;
        this->cov = b.cov;
        return *this;
    };

    StatesGroup operator+(const Matrix<double, DIM_STATE, 1> &state_add) {
        StatesGroup a;
        a.rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
        a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
        a.bias_g = this->bias_g + state_add.block<3, 1>(9, 0);
        a.bias_a = this->bias_a + state_add.block<3, 1>(12, 0);
        a.gravity = this->gravity + state_add.block<3, 1>(15, 0);
        a.cov = this->cov;
        return a;
    };

    StatesGroup &operator+=(const Matrix<double, DIM_STATE, 1> &state_add) {
        this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        this->pos_end += state_add.block<3, 1>(3, 0);
        this->vel_end += state_add.block<3, 1>(6, 0);
        this->bias_g += state_add.block<3, 1>(9, 0);
        this->bias_a += state_add.block<3, 1>(12, 0);
        this->gravity += state_add.block<3, 1>(15, 0);
        return *this;
    };

    Matrix<double, DIM_STATE, 1> operator-(const StatesGroup &b) {
        Matrix<double, DIM_STATE, 1> a;
        M3D rotd(b.rot_end.transpose() * this->rot_end);
        a.block<3, 1>(0, 0) = Log(rotd);
        a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
        a.block<3, 1>(6, 0) = this->vel_end - b.vel_end;
        a.block<3, 1>(9, 0) = this->bias_g - b.bias_g;
        a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
        a.block<3, 1>(15, 0) = this->gravity - b.gravity;
        return a;
    };

    void resetpose() {
        this->rot_end = M3D::Identity();
        this->pos_end = Zero3d;
        this->vel_end = Zero3d;
    }

    M3D rot_end;                               // the estimated attitude (rotation matrix) at the end lidar point
    V3D pos_end;                               // the estimated position at the end lidar point (world frame)
    V3D vel_end;                               // the estimated velocity at the end lidar point (world frame)
    V3D bias_g;                                // gyroscope bias
    V3D bias_a;                                // accelerator bias
    V3D gravity;                               // the estimated gravity acceleration
    Matrix<double, DIM_STATE, DIM_STATE> cov;  // states covariance
};

template <typename T>
T rad2deg(T radians) {
    return radians * 180.0 / PI_M;
}

template <typename T>
T deg2rad(T degrees) {
    return degrees * PI_M / 180.0;
}

template <typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g,
                const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R) {
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++) {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
    }
    return move(rot_kp);
}

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec:  normalized x0
*/
template <typename T>
bool esti_normvector(Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold, const int &point_num) {
    MatrixXf A(point_num, 3);
    MatrixXf b(point_num, 1);
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < point_num; j++) {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }
    normvec = A.colPivHouseholderQr().solve(b);

    for (int j = 0; j < point_num; j++) {
        if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold) {
            return false;
        }
    }

    normvec.normalize();
    return true;
}

float calc_dist(PointType p1, PointType p2) {
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

float calc_dist(pcl::PointXYZINormal p1, pcl::PointXYZINormal p2) {
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

template <typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold) {
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }

    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        // 这个abcd参数经过归一化了，所以按照点到平面方程，可以省略分母项
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold) {
            return false;
        }
    }
    return true;
}

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R) {
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr) {
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = ypr(0) / 180.0 * M_PI;
    Scalar_t p = ypr(1) / 180.0 * M_PI;
    Scalar_t r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0,
        sin(y), cos(y), 0,
        0, 0, 1;

    Eigen::Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p),
        0., 1., 0.,
        -sin(p), 0., cos(p);

    Eigen::Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0.,
        0., cos(r), -sin(r),
        0., sin(r), cos(r);

    return Rz * Ry * Rx;
}

Eigen::Matrix3d g2R(const Eigen::Vector3d &g) {
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = R2ypr(R0).x();
    R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}

// [START] FASTLIO COLOR
// For the consideration of avoiding alignment error while running, we prefer using datatype of Eigen with no alignment.
// In addition, for higher realtime performance (don't expect too much), you can modify the option with alignment, but you might carefully be aware of the crash of the program.
#define EIGEN_DATA_TYPE_DEFAULT_OPTION Eigen::DontAlign
// #define EIGEN_DATA_TYPE_DEFAULT_OPTION Eigen::AutoAlign

template <int M, int N, int option = (EIGEN_DATA_TYPE_DEFAULT_OPTION | Eigen::RowMajor)>
using eigen_mat_d = Eigen::Matrix<double, M, N, option>;

template <int M, int N, int option = (EIGEN_DATA_TYPE_DEFAULT_OPTION | Eigen::RowMajor)>
using eigen_mat_d = Eigen::Matrix<double, M, N, option>;

template <int M, int N, int option = (EIGEN_DATA_TYPE_DEFAULT_OPTION | Eigen::RowMajor)>
using eigen_mat_f = Eigen::Matrix<float, M, N, option>;

template <typename T, int M, int N, int option = (EIGEN_DATA_TYPE_DEFAULT_OPTION | Eigen::RowMajor)>
using eigen_mat_t = Eigen::Matrix<T, M, N, option>;

template <int M, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
using eigen_vec_d = Eigen::Matrix<double, M, 1, option>;

template <int M, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
using eigen_vec_f = Eigen::Matrix<float, M, 1, option>;

template <typename T, int M, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
using eigen_vec_t = Eigen::Matrix<T, M, 1, option>;

template <typename T, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
using eigen_q_t = Eigen::Quaternion<T, option>;

template <typename T>
using eigen_angleaxis_t = Eigen::AngleAxis<T>;

template <typename T, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
using eigen_pose_t = Eigen::Transform<T, 3, Eigen::Isometry, option>;

template <int M, int N>
using eigen_mat = eigen_mat_d<M, N>;

template <int M>
using eigen_vec = eigen_vec_d<M>;

typedef eigen_vec<2> vec_2;
typedef eigen_vec<3> vec_3;
typedef eigen_vec<4> vec_4;
typedef eigen_vec<6> vec_6;
typedef eigen_vec<7> vec_7;
typedef eigen_vec<12> vec_12;
typedef eigen_mat<3, 3> mat_3_3;
typedef eigen_mat<4, 4> mat_4_4;
typedef eigen_mat<6, 6> mat_6_6;
typedef eigen_mat<12, 12> mat_12;
typedef eigen_mat<6, 12> mat_6_12;
typedef eigen_mat<12, 6> mat_12_6;
typedef eigen_angleaxis_t<double> eigen_angleaxis;
typedef Eigen::Quaternion<double, EIGEN_DATA_TYPE_DEFAULT_OPTION> eigen_q;
typedef Eigen::Transform<double, 3, Eigen::Isometry, EIGEN_DATA_TYPE_DEFAULT_OPTION> eigen_pose;
typedef std::vector<eigen_q> eigen_q_vec;

// namespace Common_tools
// {

template <typename T>
inline T angle_refine(const T &rad) {
    // Refine angle to [-pi, pi]
    T rad_afr_refined = (rad - (floor(rad / T(2 * M_PI)) * T(2 * M_PI)));
    if (rad_afr_refined > T(M_PI)) {
        rad_afr_refined -= T(2 * M_PI);
    }
    return rad_afr_refined;
}

/*****
    Some operator based tools for Eigen::Vector<T>
    Example:
        a. Eigen::Vector<T> from array:                Eigen::Vector<T> << data_rhs
        b. Eigen::Vector<T> to array:                  Eigen::Vector<T> >> data_rhs
*****/
template <typename T, int M, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
inline const void operator<<(eigen_vec_t<T, M, option> &eigen_vec_lhs, const T *data_rhs) {
    for (size_t i = 0; i < M; i++) {
        eigen_vec_lhs(i) = data_rhs[i];
    }
}

template <typename T, int M, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
inline const void operator>>(const eigen_vec_t<T, M, option> &eigen_vec_lhs, T *data_rhs) {
    for (size_t i = 0; i < M; i++) {
        data_rhs[i] = eigen_vec_lhs(i);
    }
}

template <typename T, int M, typename TT = T, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
inline const void operator<<(eigen_vec_t<T, M, option> &eigen_vec_lhs, const std::pair<std::vector<TT> *, int> &std_vector_start) {
    // Loading data from a std::vector, from the starting point
    // Example: eigen_vec_lhs << std::make_pair(&std::vector, starting_point)
    for (size_t i = 0; i < M; i++) {
        eigen_vec_lhs(i) = T((*std_vector_start.first)[std_vector_start.second + i]);
    }
}

template <typename T, int M, typename TT = T, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
inline const void operator<<(const eigen_vec_t<T, M, option> &eigen_vec_lhs, std::pair<std::vector<TT> *, int> &std_vector_start) {
    for (size_t i = 0; i < M; i++) {
        (*std_vector_start.first)[std_vector_start.second + i] = TT(eigen_vec_lhs(i));
    }
}

/*****
    Some operator based tools for Eigen::Quaternion, before using these tools, make sure you are using the uniform quaternion,
    otherwise some of the unwanted results will be happend.
    Example:
        a. Quaternion from array:                   Eigen::Quaternion << data_rhs
        b. Quaternion to array:                     Eigen::Quaternion >> data_rhs
        c. Rotation angle multiply(*=) a scalar:    Eigen::Quaternion *= scalar
        d. Rotation angle multiply(*=) a scalar:    Eigen::Quaternion * scalar
*****/
template <typename T, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
inline const void operator<<(eigen_q_t<T, option> &eigen_q_lhs, const T *data_rhs) {
    eigen_q_lhs.w() = data_rhs[0];
    eigen_q_lhs.x() = data_rhs[1];
    eigen_q_lhs.y() = data_rhs[2];
    eigen_q_lhs.z() = data_rhs[3];
}

template <typename T, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
inline const void operator>>(const eigen_q_t<T, option> &eigen_q_lhs, T *data_rhs) {
    data_rhs[0] = eigen_q_lhs.w();
    data_rhs[1] = eigen_q_lhs.x();
    data_rhs[2] = eigen_q_lhs.y();
    data_rhs[3] = eigen_q_lhs.z();
}

template <typename T, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
inline const eigen_q_t<T, option> &operator*=(eigen_q_t<T, option> &eigen_q_lhs, const T &s) {
    Eigen::AngleAxis<T> angle_axis(eigen_q_lhs);
    angle_axis *= s;
    eigen_q_lhs = eigen_q_t<T, option>(angle_axis);
    return eigen_q_lhs;
}

template <typename T, int option = EIGEN_DATA_TYPE_DEFAULT_OPTION>
inline const eigen_q_t<T, option> &operator*(const eigen_q_t<T, option> &eigen_q_lhs, const T &s) {
    Eigen::AngleAxis<T> angle_axis(eigen_q_lhs);
    angle_axis *= s;
    return eigen_q_t<T, option>(angle_axis);
}

/*****
    Conversion between eigen angle_axis and data array
    Example:
        a. AngleAxis from array: Eigen::AngleAxis << data_rhs
        b. AngleAxis to array:   Eigen::AngleAxis >> data_rhs
        c. Rotation angle multiply(*=) a scalar:    Eigen::AngleAxis *= scalar
        d. Rotation angle multiply(*=) a scalar:    Eigen::AngleAxis * scalar
*****/
template <typename T>
inline const void operator<<(Eigen::AngleAxis<T> &eigen_axisangle_lhs, const T *data_rhs) {
    T vec_norm = sqrt(data_rhs[0] * data_rhs[0] + data_rhs[1] * data_rhs[1] + data_rhs[2] * data_rhs[2]);
    if (vec_norm != T(0.0)) {
        eigen_axisangle_lhs.angle() = vec_norm;
        eigen_axisangle_lhs.axis() << data_rhs[0] / vec_norm, data_rhs[1] / vec_norm, data_rhs[2] / vec_norm;
    } else {
        eigen_axisangle_lhs.angle() = vec_norm;
        eigen_axisangle_lhs.axis() << vec_norm * data_rhs[0], vec_norm * data_rhs[1], vec_norm * data_rhs[2];  // For the consideration of derivation
    }
}

template <typename T>
inline const void operator>>(const Eigen::AngleAxis<T> &eigen_axisangle_lhs, T *data_rhs) {
    T vec_norm = eigen_axisangle_lhs.angle();
    data_rhs[0] = eigen_axisangle_lhs.axis()(0) * vec_norm;
    data_rhs[1] = eigen_axisangle_lhs.axis()(1) * vec_norm;
    data_rhs[2] = eigen_axisangle_lhs.axis()(2) * vec_norm;
}

template <typename T>
inline const Eigen::AngleAxis<T> operator*=(Eigen::AngleAxis<T> &eigen_axisangle_lhs, const T &s) {
    eigen_axisangle_lhs.angle() *= s;
    return eigen_axisangle_lhs;
}

template <typename T>
inline const Eigen::AngleAxis<T> operator*(const Eigen::AngleAxis<T> &eigen_axisangle_lhs, const T &s) {
    Eigen::AngleAxis<T> angle_axis(eigen_axisangle_lhs);
    angle_axis.angle() *= s;
    return angle_axis;
}

// [END] FAST LIO COLOR

#endif