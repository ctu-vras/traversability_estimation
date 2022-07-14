APP_NAME=ml:traversability
CONTAINER_NAME=traversability
OS_VER=$(shell uname -m)

build:
	docker build -t $(APP_NAME) -f Dockerfile_${OS_VER} .

inference:
	docker run -it --rm --runtime nvidia --network host -v /mnt:/mnt $(APP_NAME) bash
	
