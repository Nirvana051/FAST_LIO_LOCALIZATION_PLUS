#include <iostream>
#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <unistd.h>
#include <condition_variable>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Int32.h>
#include <tf/transform_datatypes.h>
#include <Eigen/Geometry>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/impl/transforms.hpp>

#include "template_alignment.h"

constexpr float kMAPVOXELSIZE = 0.2;
constexpr float kLOCALMAP = 10.0;
constexpr size_t kNUMBER = 36.0;


class ReLocalization {
public:   
    
    ReLocalization(const std::string& map_path);
    
    ~ReLocalization();

    //! detach thread
    void ReleaseThread(void);

    //! load map
    bool LoadMap(void);

    //! main loop
    void Run(void);

    //! 
    bool RefinePose(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr in,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr local_map,
                    Eigen::Matrix4f& T);
    
    /**
     * @brief GICP (Generalized Iterative Closest Point) 点云精配准算法
     * @param in 输入的源点云
     * @param local_map 目标点云（局部地图）  
     * @param T 输入初始变换矩阵，输出配准结果
     * @return true 配准成功，false 配准失败
     */
    bool GicpMatch(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr in,
                   pcl::PointCloud<pcl::PointXYZINormal>::Ptr local_map,
                   Eigen::Matrix4f& T);
    
    /**
     * @brief NDT (Normal Distributions Transform) 点云粗配准算法
     * 
     * NDT算法是一种基于正态分布的点云配准方法，具有以下特点：
     * - 将目标点云离散化为3D网格，每个网格用正态分布表示
     * - 使用Newton优化方法寻找最优变换
     * - 对噪声和离群点具有良好的鲁棒性
     * - 收敛速度快，适合作为粗配准算法
     * 
     * @param in 输入的源点云（当前LiDAR扫描）
     * @param local_map 目标点云（局部地图）
     * @param T 输入初始变换矩阵，输出NDT配准结果的4x4变换矩阵
     * @return true 配准成功，false 配准失败
     */
    bool NDTMatch(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr in,
                  pcl::PointCloud<pcl::PointXYZINormal>::Ptr local_map,
                  Eigen::Matrix4f& T);

    //! relocalization for first match
    bool reLocalization(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr in,
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr local_map,
                      Eigen::Matrix4f& pose_inv);
    
    //! 
    void Odometry(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr in,
                  Eigen::Matrix4f& pos_predict);

    //! find parts map give initial pose guess
    void FindPartsMapInGlobalMap(const Eigen::Matrix4f& estimation_pose,
                                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr& local_map);

    bool Localization(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr in,
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr local_map,
                      Eigen::Matrix4f& pos_inv);

    void TestFovMap(const Eigen::Matrix4f& pos_inv, pcl::PointCloud<pcl::PointXYZINormal>::Ptr scan);

    void GetInitialPose(const geometry_msgs::PoseStamped::ConstPtr &goal) {
        // 提取位置 - RViz 2D Nav Goal只提供x,y信息
        double x = goal->pose.position.x;
        double y = goal->pose.position.y;
        double z = 1.0;  // 固定高度为0.5米
        
        // 提取四元数并转换为欧拉角 - 2D Nav Goal只有yaw信息
        tf::Quaternion q(goal->pose.orientation.x,
                        goal->pose.orientation.y,
                        goal->pose.orientation.z,
                        goal->pose.orientation.w);
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        
        // 2D Nav Goal约束：roll和pitch保持为0
        // roll = 0.0;
        // pitch = 0.0;
        
        // 打印接收到的信息用于调试
        std::cout << "Received 2D Nav Goal: x=" << x << ", y=" << y << ", yaw=" << yaw*180.0/M_PI << "°" << std::endl;

        std::cout << "Received 2D Nav Goal: roll=" << roll*180.0/M_PI << ", pitch=" << pitch*180.0/M_PI << ", yaw=" << yaw*180.0/M_PI << "°" << std::endl;
        
        pitch = - M_PI/3.0;

        // 输出修正后的姿态
        std::cout << "Corrected 2D Nav Goal: roll=" << roll*180.0/M_PI << ", pitch=" << pitch*180.0/M_PI << ", yaw=" << yaw*180.0/M_PI << "°" << std::endl;

        q = tf::createQuaternionFromRPY(roll, pitch, yaw);
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

        //转换后角度
        std::cout << "Transfored 2D Nav Goal: roll=" << roll*180.0/M_PI << ", pitch=" << pitch*180.0/M_PI << ", yaw=" << yaw*180.0/M_PI << "°" << std::endl;

        // 使用Eigen四元数转换
        Eigen::Quaternionf eigen_q(q.w(), q.x(), q.y(), q.z());
        Eigen::Matrix3f base_rotation = eigen_q.toRotationMatrix();
        
        // // 构建机器人基座的旋转矩阵（只有yaw旋转）
        // Eigen::Matrix3f base_rotation;
        // base_rotation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
        //             Eigen::AngleAxisf( 0*M_PI / 3.0, Eigen::Vector3f::UnitY()) *  // 考虑LiDAR的Y轴-60度旋转补偿
        //             Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitX());   // roll = 0
                
        // 构建完整的4x4变换矩阵
        initial_pose_ = Eigen::Matrix4f::Identity();
        initial_pose_.block<3,3>(0,0) = base_rotation;
        initial_pose_(0,3) = x;
        initial_pose_(1,3) = y;
        initial_pose_(2,3) = z;
        
        has_initial_pose_ = true;
        
        // 输出完整的变换矩阵用于调试
        std::cout << "Initial pose with LiDAR compensation (z=0.5m):\n" << initial_pose_ << std::endl;
        
        // 输出变换后的欧拉角用于验证
        Eigen::Vector3f euler = base_rotation.eulerAngles(2, 1, 0); // ZYX顺序
        std::cout << "Final orientation - yaw: " << euler[0]*180.0/M_PI 
                << "°, pitch: " << euler[1]*180.0/M_PI 
                << "°, roll: " << euler[2]*180.0/M_PI << "°" << std::endl;
    }


    //! Encoder callback function
    void EncoderCb(const std_msgs::Int32::ConstPtr &encoder_msg) {
        encoder_count_ = encoder_msg->data;
        std::cout << "Received encoder data: " << encoder_count_/encoder_mutex_*360 << std::endl;
        
        // 检查encoder_mutex_是否为零，避免除零错误
        if (encoder_mutex_ == 0) {
            encoder_mutex_ = 1.0; // 设置默认值或从参数服务器读取
        }
        
        // 1. 基座到转子的变换（固定的机械结构）
        q_base_rotor = Eigen::Quaterniond::Identity(); // 假设基座和转子重合，根据实际情况调整
        
        // 2. 转子坐标系的旋转变换（基于编码器值）
        q_rotor_rotorframe = Eigen::AngleAxisd(-45 * M_PI/180.0, Eigen::Vector3d::UnitZ()) * 
                            Eigen::AngleAxisd(encoder_count_/encoder_mutex_ * 2 * M_PI, Eigen::Vector3d::UnitZ());
        
        // 3. 转子到激光雷达的变换（固定的机械结构）
        q_rotorframe_lidar = Eigen::AngleAxisd(-60 * M_PI/180.0, Eigen::Vector3d::UnitY()); // 假设转子和激光雷达重合，根据实际情况调整
        
        // 4. 完整的变换链：基座 -> 转子 -> 转子旋转 -> 激光雷达
        // T_lidar_base = T_lidar_rotorframe * T_rotorframe_rotor * T_rotor_base
        Eigen::Quaterniond q_lidar_base = q_rotorframe_lidar.inverse() * q_rotor_rotorframe.inverse() * q_base_rotor.inverse();
        
        // 5. 构建从激光雷达到基座的齐次变换矩阵 (L2B)
        transform_L2B = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f rotation_matrix = q_lidar_base.toRotationMatrix().cast<float>();
        transform_L2B.block<3,3>(0,0) = rotation_matrix;
    
        // 平移分量（激光雷达相对于基座的位置偏移）
        transform_L2B(0,3) = 0.0; // x方向偏移
        transform_L2B(1,3) = 0.0; // y方向偏移
        transform_L2B(2,3) = 0.0; // z方向偏移（如果激光雷达有高度偏移）
    }

    //! get map
    void GetMap(pcl::PointCloud<pcl::PointXYZINormal>& map) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv.wait(lock, [this] { return map_ready_; }); 
        
        map  = *map_origin_ptr_;

        std::cout << "load origin pointcloud size = " << map.points.size() << std::endl;
    }

    void SetRelocalizationFlag() {
        relocalzation_flag_ = true;
    }

    bool GetRelocalizationFlag() {
        return relocalzation_flag_;
    }

    void SetExit() {
         exit_ = true;
    }

    Eigen::Matrix4f GetInitialPose() const {
        return initial_pose_;
    }

    Eigen::Matrix4f GetTransform_L2B() const {
        return transform_L2B;
    }


    //! rough match

    //! refine match

    //! fusion pose
    bool has_initial_pose_;
private:
    std::string map_path_;
    std::unique_ptr<std::thread> localization_thread_ptr_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr map_origin_ptr_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr map_cloud_ptr_;
    Eigen::Matrix4f initial_pose_;

    std::mutex mtx_;               
    std::condition_variable cv;
    bool map_ready_ = false;
    bool relocalzation_flag_;

    pcl::VoxelGrid<pcl::PointXYZINormal> voxel_filter_;

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> gicp_;
    pcl::IterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> icp_;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr current_scan;

    bool exit_;
    
    // Encoder related variables
    int32_t encoder_count_;
    Eigen::Quaterniond q_rotorframe_lidar; //激光雷达相对于转子坐标系的固定变换
    Eigen::Quaterniond q_rotor_rotorframe; //转子坐标系的旋转变换
    Eigen::Quaterniond q_base_rotor; //基座相对于转子的变换
    Eigen::Matrix4f transform_L2B; //激光雷达到基座坐标系的齐次变换矩阵
    float encoder_mutex_;

};