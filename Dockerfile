# Use PyTorch CUDA runtime image (with CUDA 12.6)
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    zstd \
    locales \
    software-properties-common \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository universe

# Install ROS2 
RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | grep -F "tag_name" \
    | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    rm /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Install ZED SDK
RUN wget -q -O ZED_SDK.run \
    https://download.stereolabs.com/zedsdk/5.1/cu13/ubuntu22 \
    && chmod +x ZED_SDK.run \
    && ./ZED_SDK.run -- silent  \
    && rm ZED_SDK.run

RUN cd /usr/local/zed && python3 get_python_api.py

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --no-build-isolation -r requirements.txt

RUN pip install --ignore-installed /usr/local/zed/pyzed-5.1-cp311-cp311-linux_x86_64.whl

#Default command (can be overridden)
CMD ["/bin/bash"]
