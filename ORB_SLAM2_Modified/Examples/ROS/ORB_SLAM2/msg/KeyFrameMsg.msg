# KeyFrame.msg
Header header

# 关键帧ID
uint32 keyframe_id

# 关键帧的位姿 (世界坐标系下)
geometry_msgs/Pose pose

# 关键帧的左右目图像
sensor_msgs/Image left_image
sensor_msgs/Image right_image
# [新增] 为左右目添加CameraInfo字段
sensor_msgs/CameraInfo left_info
sensor_msgs/CameraInfo right_info
