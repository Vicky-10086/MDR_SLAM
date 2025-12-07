# OpenMVS Dense Reconstruction Pipeline

This is a complete OpenMVS dense reconstruction pipeline project based on Docker. It includes a full toolchain ranging from ROS bag data extraction and COLMAP sparse reconstruction to OpenMVS texture mapping, along with a Sim3-based ground truth evaluation script.

## 🛠️ Environment Setup
This project is deployed using Docker, based on Ubuntu 22.04/24.04.

```bash
# Build the image
docker build -t my_openmvs:v1 .
