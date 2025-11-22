IMAGE_NAME ?= audio
ENV_FILE ?= .env
DOCKER_RUN_ARGS ?=

.PHONY: docker-build docker-run

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p 8000:8000 --env-file $(ENV_FILE) $(DOCKER_RUN_ARGS) $(IMAGE_NAME)
