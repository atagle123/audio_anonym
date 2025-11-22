IMAGE_NAME ?= audio
ENV_FILE ?= .env
DOCKER_RUN_ARGS ?=
CONTAINER_NAME ?= $(IMAGE_NAME)

.PHONY: docker-build docker-run docker-logs

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm --name $(CONTAINER_NAME) -p 8000:8000 --env-file $(ENV_FILE) $(DOCKER_RUN_ARGS) $(IMAGE_NAME)

docker-logs:
	docker logs -f $(CONTAINER_NAME)
