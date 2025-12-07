import open3d as o3d

# 读取你的纯点云文件
pcd = o3d.io.read_point_cloud("scene_dense.ply")
print(f"读取成功，包含 {len(pcd.points)} 个点")

# 保存为 PCD
o3d.io.write_point_cloud("final_cloud.pcd", pcd)
print("转换完成！")
