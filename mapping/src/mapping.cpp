#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <mutex>
#include <thread>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h> 

#include <ros/ros.h>
#include <ros/spinner.h>
#include <pcl_ros/transforms.h> 
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;

typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class MapBuilder {
private:
    ros::NodeHandle nh_;
    ros::Publisher pub_global_map_;
    ros::Subscriber pointcloud_sub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    
    std::mutex map_mutex_;
    PointCloud::Ptr global_map_;
    std::vector<int> global_map_counts_; 
    
    pcl::VoxelGrid<PointT> voxel_filter_;
    pcl::PassThrough<pcl::PointXYZRGB> pass_through_filter_; 
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;

    std::string world_frame_;
    double max_range_;
    double fusion_radius_sq_;
    double normal_similarity_threshold_;
    int min_update_count_for_keep_;
    int clean_map_interval_;
    bool use_confidence_fusion_;
    int frame_count_ = 0;

public:
    MapBuilder() : nh_("~") {
        std::string topic_pointcloud;
        double resolution, fusion_radius;
        
        nh_.param<std::string>("topic_pointcloud", topic_pointcloud, "/elas/point_cloud");
        nh_.param<std::string>("world_frame", world_frame_, "map");
        
        nh_.param<double>("resolution", resolution, 0.05);
        nh_.param<double>("max_range", max_range_, 50.0); 
        nh_.param<double>("fusion_radius", fusion_radius, 0.1);
        nh_.param<double>("normal_similarity_threshold", normal_similarity_threshold_, 0.8);
        
        nh_.param<int>("min_update_count_for_keep", min_update_count_for_keep_, 1); 
        nh_.param<int>("clean_map_interval", clean_map_interval_, 100);
        nh_.param<bool>("use_confidence_fusion", use_confidence_fusion_, true);

        fusion_radius_sq_ = fusion_radius * fusion_radius;

        global_map_.reset(new PointCloud());
        kdtree_.reset(new pcl::KdTreeFLANN<PointT>());

        voxel_filter_.setLeafSize(resolution, resolution, resolution);
        pass_through_filter_.setFilterFieldName("z");
        pass_through_filter_.setFilterLimits(0.1, max_range_); 

        tf_buffer_ = std::unique_ptr<tf2_ros::Buffer>(new tf2_ros::Buffer(ros::Duration(30.0))); 
        tf_listener_ = std::unique_ptr<tf2_ros::TransformListener>(new tf2_ros::TransformListener(*tf_buffer_));
        
        pointcloud_sub_ = nh_.subscribe(topic_pointcloud, 5, &MapBuilder::callback, this);
        pub_global_map_ = nh_.advertise<sensor_msgs::PointCloud2>("stitched_map", 2);
        
        ROS_INFO("Map Builder Running (Rescue Mode). Listening to %s", topic_pointcloud.c_str());
    }

    PointCloud::Ptr getSparseMap() {
        std::lock_guard<std::mutex> lock(map_mutex_);
        PointCloud::Ptr sparse_map(new PointCloud());
        if (global_map_->empty()) return sparse_map;
        voxel_filter_.setInputCloud(global_map_);
        voxel_filter_.filter(*sparse_map);
        return sparse_map;
    }

    void callback(const sensor_msgs::PointCloud2::ConstPtr& msgPointCloud) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msgPointCloud, *cloud_in);
        if (cloud_in->empty()) return;
        
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
        pass_through_filter_.setInputCloud(cloud_in);
        pass_through_filter_.filter(*cloud_filtered);
        
        if (cloud_filtered->empty()) {
             ROS_WARN_THROTTLE(2.0, "Max range filtered all points! Using raw points.");
             *cloud_filtered = *cloud_in;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
        try {
            geometry_msgs::TransformStamped transform = tf_buffer_->lookupTransform(
                world_frame_, msgPointCloud->header.frame_id, msgPointCloud->header.stamp, ros::Duration(1.0));
            Eigen::Matrix4f mat;
            pcl_ros::transformAsMatrix(transform.transform, mat);
            pcl::transformPointCloud(*cloud_filtered, *cloud_transformed_rgb, mat);
        } catch (tf2::TransformException &ex) {
            ROS_WARN_THROTTLE(1.0, "TF Error: %s", ex.what());
            return; 
        }

        PointCloud::Ptr cloud_with_normals(new PointCloud());
        cloud_with_normals->resize(cloud_transformed_rgb->size());

        for (size_t i = 0; i < cloud_transformed_rgb->size(); ++i) {
            cloud_with_normals->points[i].x = cloud_transformed_rgb->points[i].x;
            cloud_with_normals->points[i].y = cloud_transformed_rgb->points[i].y;
            cloud_with_normals->points[i].z = cloud_transformed_rgb->points[i].z;
            cloud_with_normals->points[i].rgb = cloud_transformed_rgb->points[i].rgb;
            
            cloud_with_normals->points[i].normal_x = 0;
            cloud_with_normals->points[i].normal_y = 0;
            cloud_with_normals->points[i].normal_z = 1.0; 
        }

        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        pcl::NormalEstimation<PointT, PointT> ne;
        ne.setInputCloud(cloud_with_normals);
        ne.setSearchMethod(tree);
        ne.setKSearch(10); 
        ne.compute(*cloud_with_normals);

        for (auto& p : cloud_with_normals->points) {
            if (!std::isfinite(p.normal_x) || !std::isfinite(p.normal_y) || !std::isfinite(p.normal_z)) {
                p.normal_x = 0.0; p.normal_y = 0.0; p.normal_z = 1.0;
            }
        }

        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            
            if (use_confidence_fusion_) 
            {
                if (global_map_->empty()) {
                    *global_map_ = *cloud_with_normals;
                    global_map_counts_.resize(global_map_->size(), 5); 
                } else {
                    kdtree_->setInputCloud(global_map_);
                    PointCloud::Ptr new_points(new PointCloud());
                    std::vector<int> new_counts;

                    for (const auto& p_new : cloud_with_normals->points) {
                        std::vector<int> idx(1);
                        std::vector<float> dist_sq(1);
                        kdtree_->nearestKSearch(p_new, 1, idx, dist_sq);

                        if (!idx.empty() && dist_sq[0] < fusion_radius_sq_) {
                            int i = idx[0];
                            PointT& p_old = global_map_->points[i];
                            
                            Eigen::Vector3f n1(p_old.normal_x, p_old.normal_y, p_old.normal_z);
                            Eigen::Vector3f n2(p_new.normal_x, p_new.normal_y, p_new.normal_z);
                            
                            if (n1.dot(n2) > normal_similarity_threshold_) {
                                float w = (float)global_map_counts_[i];
                                float w_sum = w + 1.0f;
                                p_old.x = (p_old.x * w + p_new.x) / w_sum;
                                p_old.y = (p_old.y * w + p_new.y) / w_sum;
                                p_old.z = (p_old.z * w + p_new.z) / w_sum;
                                if (global_map_counts_[i] < 100) global_map_counts_[i]++;
                            } else {
                                new_points->push_back(p_new);
                                new_counts.push_back(5); 
                            }
                        } else {
                            new_points->push_back(p_new);
                            new_counts.push_back(5); 
                        }
                    }
                    *global_map_ += *new_points;
                    global_map_counts_.insert(global_map_counts_.end(), new_counts.begin(), new_counts.end());
                }

                frame_count_++;
                if (min_update_count_for_keep_ > 0 && frame_count_ % clean_map_interval_ == 0) {
                    PointCloud::Ptr cleaned(new PointCloud());
                    std::vector<int> cleaned_counts;
                    cleaned->reserve(global_map_->size());
                    cleaned_counts.reserve(global_map_counts_.size());
                    
                    for (size_t i = 0; i < global_map_->size(); ++i) {
                        if (global_map_counts_[i] >= min_update_count_for_keep_) {
                            cleaned->push_back(global_map_->points[i]);
                            cleaned_counts.push_back(global_map_counts_[i]);
                        }
                    }
                    global_map_.swap(cleaned);
                    global_map_counts_.swap(cleaned_counts);
                }
            }
            else 
            {
                *global_map_ += *cloud_with_normals;
            }
        }

        sensor_msgs::PointCloud2 output;
        if (global_map_->size() < 50000) {
            pcl::toROSMsg(*global_map_, output);
        } else {
            pcl::toROSMsg(*getSparseMap(), output);
        }
        
        output.header.frame_id = world_frame_;
        output.header.stamp = ros::Time::now();
        pub_global_map_.publish(output);
        
        ROS_INFO_THROTTLE(2.0, "Map Size: %ld", global_map_->size());
    }
    
    void save(const std::string& path) {
        std::lock_guard<std::mutex> lock(map_mutex_);
        if (global_map_->empty()) return;
        pcl::io::savePCDFileBinary(path, *global_map_);
        ROS_INFO("Saved %ld points to %s", global_map_->size(), path.c_str());
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "map_builder");
    MapBuilder builder;
    ros::AsyncSpinner spinner(2);
    spinner.start();
    
    ROS_INFO("Robust Map Builder Started.");
    ros::waitForShutdown();
    
    builder.save("/root/catkin_ws/output/mapping/final_map.pcd");
    return 0;
}
