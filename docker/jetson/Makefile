APP_NAME=ml:traversability
CONTAINER_NAME=traversability

build:
	docker build -t $(APP_NAME) -f Dockerfile .

inference:
	docker run -it --rm --runtime nvidia --network host \
		-v /mnt:/mnt \
		-v /home/robot/:/home/robot \
		$(APP_NAME) bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}

