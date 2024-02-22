#!/usr/bin/bash

DOCKER_IMAGE=patent-classification
DOCKER_TAG=0.4
DOCKER_BUILDKIT=1 docker build --ssh default   -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
