/**
* FINAL MODIFIED VERSION
* This file is modified for stereo compatibility, custom KeyFrame message publishing,
* and robust message synchronization using ExactTime policy.
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h> // 使用精确时间同步
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <unistd.h>

#include "System.h"
#include "ORB_SLAM2/KeyFrameMsg.h"

using namespace std;

class ImageGrabber
{
public:
    ORB_SLAM2::System* mpSLAM;
    ros::NodeHandle nh;
    
    ros::Publisher pub_pose_;
    ros::Publisher pub_odom_;
    ros::Publisher pub_path_;
    nav_msgs::Path camera_path_;

    ros::Publisher pub_keyframe_;
    
    tf::TransformBroadcaster tf_broadcaster_;
    string map_frame_id_;
    string base_frame_id_;

    bool do_rectify;
    cv::Mat M1l, M2l, M1r, M2r;

    ImageGrabber(ORB_SLAM2::System* pSLAM) : mpSLAM(pSLAM), nh("~")
    {
        pub_pose_ = nh.advertise<geometry_msgs::PoseStamped>("CameraPose", 10); 
        pub_odom_ = nh.advertise<nav_msgs::Odometry>("Odometry", 10); 
        pub_path_ = nh.advertise<nav_msgs::Path>("Path", 10); 
        pub_keyframe_ = nh.advertise<ORB_SLAM2::KeyFrameMsg>("/orb_slam2/keyframe", 10);

        nh.param<string>("map_frame_id", map_frame_id_, "map");
        nh.param<string>("base_frame_id", base_frame_id_, "camera_link");
        ROS_INFO_STREAM("TF frames configured: map_frame='" << map_frame_id_ << "', base_frame='" << base_frame_id_ << "'");
    }

    void GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft, const sensor_msgs::ImageConstPtr& msgRight,
                    const sensor_msgs::CameraInfoConstPtr& msgLeftInfo, const sensor_msgs::CameraInfoConstPtr& msgRightInfo);
};

void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft, const sensor_msgs::ImageConstPtr& msgRight,
                              const sensor_msgs::CameraInfoConstPtr& msgLeftInfo, const sensor_msgs::CameraInfoConstPtr& msgRightInfo)
{
    // [调试断点1] 在函数入口打印时间戳，确认回调是否被触发
    // ROS_INFO_STREAM("GrabStereo Callback triggered with timestamp: " << msgLeft->header.stamp);
    cv_bridge::CvImageConstPtr cv_ptrLeft, cv_ptrRight;
    try {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    cv::Mat Tcw;
    bool isKeyFrame = false;
    
    cv::Mat imLeft, imRight;

    if(do_rectify) {
        cv::remap(cv_ptrLeft->image, imLeft, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(cv_ptrRight->image, imRight, M1r, M2r, cv::INTER_LINEAR);
    } else {
        imLeft = cv_ptrLeft->image;
        imRight = cv_ptrRight->image;
    }
    
    Tcw = mpSLAM->TrackStereo(imLeft, imRight, cv_ptrLeft->header.stamp.toSec(), isKeyFrame);
 
    if (!Tcw.empty()) {
        cv::Mat Twc = Tcw.inv();
        cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
        cv::Mat twc = Twc.rowRange(0, 3).col(3);
        
        Eigen::Matrix3d eigMat;
        eigMat << Rwc.at<float>(0,0), Rwc.at<float>(0,1), Rwc.at<float>(0,2),
                  Rwc.at<float>(1,0), Rwc.at<float>(1,1), Rwc.at<float>(1,2),
                  Rwc.at<float>(2,0), Rwc.at<float>(2,1), Rwc.at<float>(2,2);
        Eigen::Quaterniond q(eigMat);

        tf::Transform transform;
        transform.setOrigin(tf::Vector3(twc.at<float>(0), twc.at<float>(1), twc.at<float>(2)));
        transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
        tf_broadcaster_.sendTransform(tf::StampedTransform(transform, msgLeft->header.stamp, map_frame_id_, base_frame_id_));

        std_msgs::Header header;
        header.stamp = msgLeft->header.stamp;
        header.frame_id = map_frame_id_;

        geometry_msgs::PoseStamped pose_msg; 
        pose_msg.header = header;
        pose_msg.pose.position.x = twc.at<float>(0);
        pose_msg.pose.position.y = twc.at<float>(1);
        pose_msg.pose.position.z = twc.at<float>(2);
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();
        pub_pose_.publish(pose_msg);
        
        nav_msgs::Odometry odom_msg;
        odom_msg.header = header;
        odom_msg.child_frame_id = base_frame_id_;
        odom_msg.pose.pose = pose_msg.pose;
        pub_odom_.publish(odom_msg);
        
        camera_path_.header = header;
        camera_path_.poses.push_back(pose_msg);     
        pub_path_.publish(camera_path_);

        if(isKeyFrame)
	    {
	        ROS_INFO("!!! NEW KEYFRAME CREATED at frame %d !!!", msgLeft->header.seq);
	        ORB_SLAM2::KeyFrameMsg kf_msg;
	        kf_msg.header = msgLeft->header;
	        kf_msg.header.frame_id = base_frame_id_;
	        kf_msg.pose = pose_msg.pose;
	        kf_msg.left_image = *msgLeft;
	        kf_msg.right_image = *msgRight;
            kf_msg.left_info = *msgLeftInfo;
            kf_msg.right_info = *msgRightInfo;
	        pub_keyframe_.publish(kf_msg);
	    }

    } else {
        ROS_WARN("Tracking failed, Tcw is empty.");
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "orb_slam2_stereo");
    ros::start();

    if(argc < 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 Stereo path_to_vocabulary path_to_settings [do_rectify(true/false)]" << endl;
        ros::shutdown();
        return 1;
    }    

    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::STEREO, true);
    ImageGrabber igb(&SLAM);

    bool do_rectify = false;
    if(argc == 4)
    {
        stringstream ss(argv[3]);
        ss >> boolalpha >> do_rectify;
    }
    igb.do_rectify = do_rectify;

    if(igb.do_rectify)
    {      
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        if(!fsSettings.isOpened()) {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;
        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;
        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;
        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;
        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
                rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0) {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }
        cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,igb.M1l,igb.M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,igb.M1r,igb.M2r);
    }

    ros::NodeHandle nh;

    // 使用通用话题名，保持代码普适性
    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "camera/left/image_rect", 10);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "camera/right/image_rect", 10);
    message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub(nh, "camera/left/camera_info", 10);
    message_filters::Subscriber<sensor_msgs::CameraInfo> right_info_sub(nh, "camera/right/camera_info", 10);

    // 使用精确时间同步策略
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> SyncPolicy;
    
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), left_sub, right_sub, left_info_sub, right_info_sub);
    
    sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2, _3, _4));

    ros::spin();
    SLAM.Shutdown();
    SLAM.SaveTrajectoryTUM("/root/catkin_ws/output/orbslam2/FrameTrajectory_TUM_Format.txt");
    SLAM.SaveTrajectoryKITTI("/root/catkin_ws/output/orbslam2/FrameTrajectory_KITTI_Format.txt");
    ros::shutdown();
 
    return 0;
}