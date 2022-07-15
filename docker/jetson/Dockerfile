FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3


RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libturbojpeg \
    software-properties-common \
    libboost-all-dev \
    libssl-dev \
    libgeos-dev \
    wget \
    nano \
    sudo \
    python3-matplotlib \
    python3-opencv \
    python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN pip3 install \
    tqdm==4.62.3 \
    yacs==0.1.6 \
    #open3d==0.11.2 \
    gnupg==2.3.1 \
    configparser==5.2.0 \
    psutil==5.8.0 \
    rospkg \
    empy

COPY ./install_ros_melodic.sh /tmp/install_ros.sh
RUN /tmp/install_ros.sh

