IMAGE=nervana-dockrepo01.fm.intel.com:5001/tarek_ssd_model_exploration:test_15

.PHONY: build run

BUILD_ARGUMENTS=
RUN_ARGUMENTS=
ifdef http_proxy
	BUILD_ARGUMENTS+=--build-arg http_proxy=$(http_proxy)
	RUN_ARGUMENTS+=--env http_proxy=$(http_proxy)
endif

ifdef https_proxy
	BUILD_ARGUMENTS+=--build-arg https_proxy=$(https_proxy)
	RUN_ARGUMENTS+=--env https_proxy=$(https_proxy)
endif

CONTEXT = $(shell pwd)

# use nvidia-docker if it is available
DOCKER := $(shell command -v nvidia-docker 2> /dev/null)
ifndef DOCKER
	DOCKER = docker
endif

# in some environments you can just mount the context, in ai-lab however
# we need to mount that nfs share directly
RUN_ARGUMENTS += -v ${CONTEXT}:/root/src
# Does not apply to Kubernetes
# docker volume create --name experiments --opt type=nfs --opt device=:/nfs/site/home/takeller/repo/ai-lab-kubernetes/examples --opt o=addr=fmcfs05n02b-03.fm.intel.com
# RUN_ARGUMENTS+=-v examples:/root/src/examples

clean:
	@rm -f .*.swp .*.swo
	@rm -f *.pyc

build: 
	${DOCKER} build -f=Dockerfile -t=${IMAGE} ${BUILD_ARGUMENTS} - < Dockerfile

push:
	${DOCKER} push ${IMAGE}

run: build_rnn
	${DOCKER} run ${RUN_ARGUMENTS} --rm -it ${IMAGE}

shell: build_rnn
	${DOCKER} run ${RUN_ARGUMENTS} --rm -it ${IMAGE} /bin/bash

# TODO? make run_k8s; make shell_k8s ?
