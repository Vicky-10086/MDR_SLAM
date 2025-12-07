# 1. Use the official Ubuntu 24.04 (Best match for the latest OpenMVS)
FROM ubuntu:24.04

# 2. Replace with Aliyun mirrors (Optimized for mainland China network speed)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources

# 3. Install dependencies
# Note: Ubuntu 24.04 supports installing libceres-dev directly via apt, avoiding manual compilation.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git mercurial \
    libpng-dev libjpeg-dev libtiff-dev libglu1-mesa-dev \
    libboost-iostreams-dev libboost-program-options-dev libboost-system-dev \
    libboost-serialization-dev libopencv-dev libcgal-dev libcgal-qt5-dev \
    libatlas-base-dev libsuitesparse-dev \
    libceres-dev \
    curl wget vim \
    && rm -rf /var/lib/apt/lists/*

# 4. Prepare working directory for dependencies
WORKDIR /deps

# 5. Install VCGLib (Required geometry library)
# Using gitclone.com proxy for faster access in China
RUN git clone https://gitclone.com/github.com/cdcseacave/VCG.git vcglib

# 6. Install OpenMVS
WORKDIR /src
RUN git clone https://gitclone.com/github.com/cdcseacave/openMVS.git openMVS && \
    mkdir openMVS_build && cd openMVS_build && \
    cmake . ../openMVS \
    -DCMAKE_BUILD_TYPE=Release \
    -DVCG_ROOT=/deps/vcglib \
    -DOpenMVS_USE_CUDA=OFF \
    -DOpenMVS_BUILD_PYTHON=OFF \
    && \
    make -j$(nproc) && \
    make install

# 7. Setup environment variables
ENV PATH="/usr/local/bin/OpenMVS:${PATH}"
WORKDIR /data

# 8. Verify installation (Prints help message to ensure success)
RUN DensifyPointCloud --help
