#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <libelas/elas.h>
#include <Eigen/Geometry>

#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <opencv2/imgproc.hpp>

#include <ORB_SLAM2/KeyFrameMsg.h>

class ElasTemporalFilter
{
public:
    typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ApproximateSyncPolicy;
    typedef message_filters::Synchronizer<ApproximateSyncPolicy> Synchronizer;

private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    ros::Subscriber sub_keyframe_;
    image_transport::SubscriberFilter sub_l_img_, sub_r_img_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_l_info_, sub_r_info_;
    boost::shared_ptr<Synchronizer> sync_;

    ros::Publisher pub_pointcloud_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    std::unique_ptr<Elas> elas_;
    boost::scoped_ptr<Elas::parameters> param_;

    bool has_active_keyframe_ = false;
    ORB_SLAM2::KeyFrameMsg current_kf_msg_;
    
    cv::Mat kf_depth_map_;
    cv::Mat kf_variance_map_;
    cv::Mat kf_stability_map_;

    std::string world_frame_ = "map";
    float initial_variance_;
    float measurement_variance_;
    float variance_threshold_;
    int stability_threshold_;
    bool use_temporal_filter_;

public:
    ElasTemporalFilter(const std::string& transport) : nh_(), private_nh_("~")
    {
        param_.reset(new Elas::parameters);
        private_nh_.param<int>("disp_min", param_->disp_min, 0);
        private_nh_.param<int>("disp_max", param_->disp_max, 255);
        private_nh_.param<float>("support_threshold", param_->support_threshold, 0.85f);
        private_nh_.param<int>("support_texture", param_->support_texture, 10);
        private_nh_.param<int>("candidate_stepsize", param_->candidate_stepsize, 5);
        private_nh_.param<int>("incon_window_size", param_->incon_window_size, 5);
        private_nh_.param<int>("incon_threshold", param_->incon_threshold, 5);
        private_nh_.param<int>("incon_min_support", param_->incon_min_support, 5);
        private_nh_.param<bool>("add_corners", param_->add_corners, false);
        private_nh_.param<int>("grid_size", param_->grid_size, 20);
        private_nh_.param<float>("beta", param_->beta, 0.02f);
        private_nh_.param<float>("gamma", param_->gamma, 3.0f);
        private_nh_.param<float>("sigma", param_->sigma, 1.0f);
        private_nh_.param<float>("sradius", param_->sradius, 2.0f);
        private_nh_.param<int>("match_texture", param_->match_texture, 1);
        private_nh_.param<int>("lr_threshold", param_->lr_threshold, 2);
        private_nh_.param<float>("speckle_sim_threshold", param_->speckle_sim_threshold, 1.0f);
        private_nh_.param<int>("speckle_size", param_->speckle_size, 200);
        private_nh_.param<int>("ipol_gap_width", param_->ipol_gap_width, 300);
        private_nh_.param<bool>("filter_median", param_->filter_median, false);
        private_nh_.param<bool>("filter_adaptive_mean", param_->filter_adaptive_mean, true);
        private_nh_.param<bool>("postprocess_only_left", param_->postprocess_only_left, true);
        private_nh_.param<bool>("subsampling", param_->subsampling, false);
        
        elas_.reset(new Elas(*param_));

        private_nh_.param<float>("initial_variance", initial_variance_, 5.0f);
        private_nh_.param<float>("measurement_variance", measurement_variance_, 0.5f);
        private_nh_.param<float>("variance_threshold", variance_threshold_, 1.0f);
        private_nh_.param<int>("stability_threshold", stability_threshold_, 1);
        private_nh_.param<std::string>("world_frame", world_frame_, "map");
        private_nh_.param<bool>("use_temporal_filter", use_temporal_filter_, true);
        
        if (use_temporal_filter_) {
            ROS_INFO("--- [MDR-SLAM Node 2] Enhanced Temporal Filter ENABLED ---");
        } else {
            ROS_WARN("--- [MDR-SLAM Node 2] ELAS-only Baseline Mode ---");
        }

        pub_pointcloud_ = private_nh_.advertise<sensor_msgs::PointCloud2>("point_cloud", 2);
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

        sub_keyframe_ = nh_.subscribe("/orb_slam2/keyframe", 5, &ElasTemporalFilter::keyframeCallback, this);

        int queue_size = 10;
        image_transport::ImageTransport it(nh_);
        sub_l_img_.subscribe(it, "stereo/left/image", queue_size, transport);
        sub_r_img_.subscribe(it, "stereo/right/image", queue_size, transport);
        sub_l_info_.subscribe(nh_, "stereo/left/camera_info", queue_size);
        sub_r_info_.subscribe(nh_, "stereo/right/camera_info", queue_size);

        if (use_temporal_filter_)
        {
            sync_.reset(new Synchronizer(ApproximateSyncPolicy(queue_size), sub_l_img_, sub_r_img_, sub_l_info_, sub_r_info_);
            sync_->registerCallback(boost::bind(&ElasTemporalFilter::ordinaryFrameCallback, this, _1, _2, _3, _4));
        }
        
        ROS_INFO("ElasTemporalFilter Initialized.");
    }

    void keyframeCallback(const ORB_SLAM2::KeyFrameMsg::ConstPtr& kf_msg)
    {
        if (use_temporal_filter_) 
        {
            if (has_active_keyframe_) {
                finalizeAndPublish(); 
            }
            initializeNewKeyframe(*kf_msg);
        }
        else
        {
            cv::Mat depth_map;
            runElas(kf_msg->left_image, kf_msg->right_image, 
                    kf_msg->left_info, kf_msg->right_info, 
                    depth_map);

            if (!depth_map.empty()) {
                createAndPublishPointCloud(depth_map, *kf_msg);
            }
        }
    }

    void ordinaryFrameCallback(const sensor_msgs::ImageConstPtr& l_img_msg, const sensor_msgs::ImageConstPtr& r_img_msg,
                                const sensor_msgs::CameraInfoConstPtr& l_info_msg, const sensor_msgs::CameraInfoConstPtr& r_info_msg)
    {
        if (!has_active_keyframe_) return; 
        updateKeyframeModel(l_img_msg, r_img_msg, l_info_msg, r_info_msg);
    }

private:
    void initializeNewKeyframe(const ORB_SLAM2::KeyFrameMsg& kf_msg) {
        current_kf_msg_ = kf_msg;
        cv::Mat initial_depth;
        
        runElas(kf_msg.left_image, kf_msg.right_image, 
                kf_msg.left_info, kf_msg.right_info, 
                initial_depth);

        if (initial_depth.empty()) {
            has_active_keyframe_ = false;
            return;
        }

        kf_depth_map_ = initial_depth.clone();
        kf_variance_map_ = cv::Mat(kf_depth_map_.size(), CV_32FC1, cv::Scalar(initial_variance_));
        kf_stability_map_ = cv::Mat(kf_depth_map_.size(), CV_8UC1, cv::Scalar(1));

        cv::Mat no_depth_mask;
        cv::compare(kf_depth_map_, 0.0f, no_depth_mask, cv::CMP_EQ);
        kf_variance_map_.setTo(std::numeric_limits<float>::infinity(), no_depth_mask);
        kf_stability_map_.setTo(0, no_depth_mask);
        
        has_active_keyframe_ = true;
    }
    
    void updateKeyframeModel(const sensor_msgs::ImageConstPtr& l_img_msg, const sensor_msgs::ImageConstPtr& r_img_msg,
                             const sensor_msgs::CameraInfoConstPtr& l_info_msg, const sensor_msgs::CameraInfoConstPtr& r_info_msg) {
        
        Eigen::Isometry3d T_world_current;
        try {
            geometry_msgs::TransformStamped tf_stamped = tf_buffer_->lookupTransform(
                world_frame_, current_kf_msg_.header.frame_id, l_img_msg->header.stamp, ros::Duration(0.1));
            T_world_current = tf2::transformToEigen(tf_stamped);
        } catch (tf2::TransformException &ex) {
            return;
        }

        Eigen::Isometry3d T_world_kf;
        tf2::fromMsg(current_kf_msg_.pose, T_world_kf);
        Eigen::Isometry3d T_current_kf = T_world_current.inverse() * T_world_kf;

        cv::Mat measured_depth;
        runElas(*l_img_msg, *r_img_msg, *l_info_msg, *r_info_msg, measured_depth);
        if(measured_depth.empty()) return;
        
        image_geometry::StereoCameraModel current_stereo_model, kf_stereo_model;
        current_stereo_model.fromCameraInfo(l_info_msg, r_info_msg);
        kf_stereo_model.fromCameraInfo(current_kf_msg_.left_info, current_kf_msg_.right_info);
        const image_geometry::PinholeCameraModel& kf_left_cam_model = kf_stereo_model.left();

        for (int v_kf = 0; v_kf < kf_depth_map_.rows; ++v_kf) {
            for (int u_kf = 0; u_kf < kf_depth_map_.cols; ++u_kf) {
                float d_kf = kf_depth_map_.at<float>(v_kf, u_kf);
                if (d_kf <= 0.0f || std::isinf(d_kf)) continue;

                cv::Point3d p3d_kf_ray = kf_left_cam_model.projectPixelTo3dRay(cv::Point2d(u_kf, v_kf));
                cv::Point3d p3d_kf(p3d_kf_ray.x * d_kf, p3d_kf_ray.y * d_kf, p3d_kf_ray.z * d_kf);

                Eigen::Vector3d p_kf_eig(p3d_kf.x, p3d_kf.y, p3d_kf.z);
                Eigen::Vector3d p_curr_eig = T_current_kf * p_kf_eig;
                
                cv::Point2d p2d_curr = current_stereo_model.left().project3dToPixel(cv::Point3d(p_curr_eig.x(), p_curr_eig.y(), p_curr_eig.z()));
                
                int u_curr = cvRound(p2d_curr.x);
                int v_curr = cvRound(p2d_curr.y);

                if (u_curr >= 0 && u_curr < measured_depth.cols && v_curr >= 0 && v_curr < measured_depth.rows) {
                    float d_meas = measured_depth.at<float>(v_curr, u_curr);
                    float d_proj = p_curr_eig.z(); 

                    float dynamic_err_tol = 0.02f * d_proj + 0.05f;

                    if (d_meas > 0.0f && std::fabs(d_proj - d_meas) < dynamic_err_tol) {
                        float& depth = kf_depth_map_.at<float>(v_kf, u_kf);
                        float& variance = kf_variance_map_.at<float>(v_kf, u_kf);
                        
                        float dynamic_meas_var = measurement_variance_ * (1.0f + (d_meas * d_meas) / 50.0f);

                        float kalman_gain = variance / (variance + dynamic_meas_var);
                        depth = depth + kalman_gain * (d_meas - depth);
                        variance = (1.0f - kalman_gain) * variance;

                        uint8_t& count = kf_stability_map_.at<uint8_t>(v_kf, u_kf);
                        if(count < 250) count++; 

                    } else {
                         uint8_t& count = kf_stability_map_.at<uint8_t>(v_kf, u_kf);
                         if (count > 1) count -= 2; 
                         else count = 0;
                    }
                } else {
                    uint8_t& count = kf_stability_map_.at<uint8_t>(v_kf, u_kf);
                    if (count > 0) count--;
                }
            }
        }
    }
    
    void finalizeAndPublish() {
        if (kf_depth_map_.empty()) return;

        PointCloud::Ptr cloud(new PointCloud());
        cv_bridge::CvImageConstPtr cv_ptr_color;
        try {
            sensor_msgs::ImageConstPtr left_image_ptr(new sensor_msgs::Image(current_kf_msg_.left_image));
            cv_ptr_color = cv_bridge::toCvShare(left_image_ptr, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        
        image_geometry::StereoCameraModel kf_stereo_model;
        kf_stereo_model.fromCameraInfo(current_kf_msg_.left_info, current_kf_msg_.right_info);

        for (int v = 0; v < kf_depth_map_.rows; ++v) {
            for (int u = 0; u < kf_depth_map_.cols; ++u) {
                
                float variance = kf_variance_map_.at<float>(v, u);
                uint8_t stability = kf_stability_map_.at<uint8_t>(v, u);
                
                if (variance < variance_threshold_ && stability >= stability_threshold_) {
                    
                    float depth = kf_depth_map_.at<float>(v, u);
                    if (depth <= 0.1f) continue;
                    
                    cv::Point3d point;
                    kf_stereo_model.projectDisparityTo3d(cv::Point2d(u, v), (kf_stereo_model.baseline() * kf_stereo_model.left().fx()) / depth, point);

                    if (point.z > 20.0 || point.z < 0.3) continue;

                    pcl::PointXYZRGB p;
                    p.x = point.x;
                    p.y = point.y;
                    p.z = point.z;
                    
                    cv::Vec3b color = cv_ptr_color->image.at<cv::Vec3b>(v, u);
                    p.b = color[0];
                    p.g = color[1];
                    p.r = color[2];
                    
                    cloud->push_back(p);
                }
            }
        }

        if (cloud->empty()) return;
        
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header = current_kf_msg_.header;
        pub_pointcloud_.publish(cloud_msg);
        
        ROS_INFO("Finalized KF %d, published point cloud with %ld points.", current_kf_msg_.header.seq, cloud->size());
    }
    
    void runElas(const sensor_msgs::Image& left_image_msg, const sensor_msgs::Image& right_image_msg,
                 const sensor_msgs::CameraInfo& left_info_msg, const sensor_msgs::CameraInfo& right_info_msg,
                 cv::Mat& depth_map) {
        
        cv_bridge::CvImageConstPtr l_cv_ptr, r_cv_ptr;
        try {
            sensor_msgs::ImageConstPtr left_image_ptr(new sensor_msgs::Image(left_image_msg));
            sensor_msgs::ImageConstPtr right_image_ptr(new sensor_msgs::Image(right_image_msg));
            l_cv_ptr = cv_bridge::toCvShare(left_image_ptr, sensor_msgs::image_encodings::MONO8);
            r_cv_ptr = cv_bridge::toCvShare(right_image_ptr, sensor_msgs::image_encodings::MONO8);
        } catch (cv_bridge::Exception& e) {
            depth_map = cv::Mat();
            return;
        }

        const int32_t width = l_cv_ptr->image.cols;
        const int32_t height = l_cv_ptr->image.rows;
        const int32_t dims[3] = {width, height, static_cast<int32_t>(l_cv_ptr->image.step[0])};

        cv::Mat disp_left(height, width, CV_32FC1);
        cv::Mat disp_right(height, width, CV_32FC1);

        elas_->process(l_cv_ptr->image.data, r_cv_ptr->image.data, disp_left.ptr<float>(), disp_right.ptr<float>(), dims);

        image_geometry::StereoCameraModel temp_model;
        temp_model.fromCameraInfo(left_info_msg, right_info_msg);
        float baseline = temp_model.baseline();
        float fx = temp_model.left().fx();

        depth_map = cv::Mat(height, width, CV_32FC1, cv::Scalar(0.0f));
        
        float lr_threshold = (float)param_->lr_threshold;

        for (int v = 0; v < height; ++v) {
            for (int u = 0; u < width; ++u) {
                float d_left = disp_left.at<float>(v, u);
                
                if (d_left <= param_->disp_min) continue;
    
                int u_right = u - (int)d_left;
                
                if (u_right >= 0 && u_right < width) {
                    float d_right = disp_right.at<float>(v, u_right);
                    
                    if (std::abs(d_left - d_right) > lr_threshold) {
                        continue;
                    }
                    
                    depth_map.at<float>(v, u) = (baseline * fx) / d_left;
                }
            }
        }
    }

    void createAndPublishPointCloud(const cv::Mat& depth_map, const ORB_SLAM2::KeyFrameMsg& kf_msg) {
        PointCloud::Ptr cloud(new PointCloud());
        cv_bridge::CvImageConstPtr cv_ptr_color;
        try {
            sensor_msgs::ImageConstPtr left_image_ptr(new sensor_msgs::Image(kf_msg.left_image));
            cv_ptr_color = cv_bridge::toCvShare(left_image_ptr, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            return;
        }

        image_geometry::StereoCameraModel kf_stereo_model;
        kf_stereo_model.fromCameraInfo(kf_msg.left_info, kf_msg.right_info);
        const image_geometry::PinholeCameraModel& kf_left_cam_model = kf_stereo_model.left();

        for (int v = 0; v < depth_map.rows; ++v) {
            for (int u = 0; u < depth_map.cols; ++u) {
                float depth = depth_map.at<float>(v, u);
                if (depth > 0.1f) { 
                    cv::Point3d p3d_kf_ray = kf_left_cam_model.projectPixelTo3dRay(cv::Point2d(u, v));
                    cv::Point3d p3d_kf(p3d_kf_ray.x * depth, p3d_kf_ray.y * depth, p3d_kf_ray.z * depth);

                    pcl::PointXYZRGB p;
                    p.x = p3d_kf.x; p.y = p3d_kf.y; p.z = p3d_kf.z;
                    cv::Vec3b color = cv_ptr_color->image.at<cv::Vec3b>(v, u);
                    p.b = color[0]; p.g = color[1]; p.r = color[2];
                    cloud->push_back(p);
                }
            }
        }

        if (cloud->empty()) return;
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header = kf_msg.header; 
        pub_pointcloud_.publish(cloud_msg);
        ROS_INFO("Published point cloud for KF %d with %ld points.", kf_msg.header.seq, cloud->size());
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "elas_ros");
    std::string transport = "raw";
    ElasTemporalFilter filter(transport);
    ros::spin();
    return 0;
}
