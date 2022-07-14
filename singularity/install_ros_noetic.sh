#!/bin/bash

sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# https://gist.github.com/Pyrestone/ef683aec160825eee5c252f22218ddb2
apt-get update
apt-get install python3-rosdep python3-rosinstall-generator python3-vcstool build-essential python3-empy libconsole-bridge-dev libpoco-dev libtinyxml-dev qtbase5-dev -y

rosdep init
rosdep update

mkdir -p /opt/ros/ros_catkin_ws
cd /opt/ros/ros_catkin_ws && \
	rosinstall_generator robot perception --rosdistro noetic --deps --tar > noetic-robot-perception.rosinstall

mkdir -p /opt/ros/ros_catkin_ws/src && \
	cd /opt/ros/ros_catkin_ws/ && \
	vcs import --input noetic-robot-perception.rosinstall /opt/ros/ros_catkin_ws/src && \
        rosdep install --from-paths /opt/ros/ros_catkin_ws/src --ignore-packages-from-source --rosdistro noetic -y && \
	/opt/ros/ros_catkin_ws/src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3

source /opt/ros/ros_catkin_ws/install_isolated/setup.bash

