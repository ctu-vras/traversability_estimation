## Docker

First of all install Docker:
```
sudo apt install docker.io
```
After that install [nvidia-docker v2.0](<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)>):
```
sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd
``` 

### Docker image with `pytorch` and `ROS melodic` on Jetson

```bash
git checkout jetson
cd ../traversability_estimation/docker/jetson/
```

Build docker image:
```bash
make build
```

Run docker container:
```bash
make inference
```
