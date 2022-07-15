APP_NAME=ml:traversability
CONTAINER_NAME=traversability

build:
	docker build -t $(APP_NAME) -f Dockerfile .

inference:
	docker run -it --rm --runtime nvidia --network host -v /mnt:/mnt $(APP_NAME) bash
	
