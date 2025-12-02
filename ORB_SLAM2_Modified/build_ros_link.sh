echo "Building ROS nodes"
cd ORB_SLAM2-1.0.0/Examples/ROS/ORB_SLAM2

# 绕过rosbuild的目录检查
sed -i 's/_rosbuild_check_package_location/#_rosbuild_check_package_location/' \
    /opt/ros/melodic/share/ros/core/rosbuild/public.cmake

# 强制清理并重建
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DROS_BUILD_TYPE=Release
make -j$(nproc)
echo "=== 构建完成 ==="  
