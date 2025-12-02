import numpy as np
import cv2

# ================= 1. 输入来自您的 yaml 文件的原始数据 =================
# 左相机 (cam1)
K_L = np.array([[532.8016818135839, 0.0, 661.4516698422946],
                [0.0, 535.4438466968883, 331.8424063682838],
                [0.0, 0.0, 1.0]])
D_L = np.array([-0.17095682206424107, 0.14277924810174208, -0.0027615426006642155, 0.004164179302936691])

# 右相机 (cam2)
K_R = np.array([[534.1398085783596, 0.0, 659.8626969186294],
                [0.0, 537.0196869213199, 347.20737943802817],
                [0.0, 0.0, 1.0]])
D_R = np.array([-0.08875825395597844, 0.05156089898455751, -0.0076689720416879565, -0.0012350125863822314])

# 左右相机之间的变换矩阵 T (来自 zed_cam2_calib.yaml 的 cam2 T_cn_cnm1)
# 注意：这是从 Right 到 Left 的变换
R = np.array([[ 0.9999897001533242, 0.0032597957040378495, 0.0031580562425951295],
              [ -0.0031925406230822622, 0.9997728570288695, -0.021072304873941128],
              [ -0.0032260303212194408, 0.02106200560958737, 0.9997729665519403]])
T = np.array([-0.12004733505991869, 0.0006825963115744139, 0.0024685752390625933])

# 图像尺寸
size = (1280, 720)

# ================= 2. 计算立体校正 =================
# 这会计算出使得左右图像行对齐的旋转矩阵 R1, R2 和投影矩阵 P1, P2
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1=K_L, distCoeffs1=D_L,
    cameraMatrix2=K_R, distCoeffs2=D_R,
    imageSize=size, R=R, T=T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0 # alpha=0 意味着裁剪掉无效的黑色区域，alpha=1 意味着保留所有像素
)

# ================= 3. 打印结果供您复制到 YAML =================
def print_yaml_format(name, mat, rows, cols):
    print(f"{name}: !!opencv-matrix")
    print(f"   rows: {rows}")
    print(f"   cols: {cols}")
    print("   dt: f")
    # 展平并格式化数据
    data_str = ", ".join([f"{x:.8f}" for x in mat.flatten()])
    print(f"   data: [ {data_str} ]")

print("# === 请用以下数据替换 NUFR_ZED_ORBSLAM.yaml 底部的内容 ===\n")

print("# 左相机校正参数")
print_yaml_format("LEFT.R", R1, 3, 3)
print_yaml_format("LEFT.P", P1, 3, 4)
print("\n# 右相机校正参数")
print_yaml_format("RIGHT.R", R2, 3, 3)
print_yaml_format("RIGHT.P", P2, 3, 4)