## Installation

Prerequisite: install [ROS](http://wiki.ros.org/ROS/Installation).

If you want to use only semantic cloud segmentation node just build the package in a catkin workspace, for example:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/ctu-vras/traversability_estimation
cd ~/catkin_ws/
catkin build traversability_estimation
```

In case you would like to run geometric cloud segmentation, traversability fusion or image segementation to point cloud projection nodes,
please follow the extended proceedure (requires access to another repositories):

- Install [vcstool](http://wiki.ros.org/vcstool) for workspace creation:
    ```bash
    sudo apt install python3-vcstool
    ```
- Create and build ROS workspace:
  ```bash
  cd ~/catkin_ws/
  vcs import src < src/traversability_estimation/config/workspace.repos
  catkin config -DCMAKE_BUILD_TYPE=Release
  catkin build
  ```
