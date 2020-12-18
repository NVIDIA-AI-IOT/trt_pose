#!/usr/bin/env bash
#
# This script builds the jetson-inference docker container from source.
# It should be run from the root dir of the jetson-inference project:
#
#     $ cd /path/to/your/trt_pose
#     $ docker/build.sh
#
# Also you should set your docker default-runtime to nvidia:
#     $ ./set_nvidia_runtime.sh
#

BASE_IMAGE=$1

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# find L4T_VERSION
source ${SCRIPTPATH}/../tools/l4t-version.sh

if [ -z $BASE_IMAGE ]; then
	if [ $L4T_VERSION = "32.4.4" ]; then
		BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3"
	elif [ $L4T_VERSION = "32.4.3" ]; then
		BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3"
	elif [ $L4T_VERSION = "32.4.2" ]; then
		BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.4.2-pth1.5-py3"
	else
		echo "cannot build jetson-inference docker container for L4T R$L4T_VERSION"
		echo "please upgrade to the latest JetPack, or build jetson-inference natively"
		exit 1
	fi
fi


echo "BASE_IMAGE=$BASE_IMAGE"
echo "TAG=trt_pose:r$L4T_VERSION"


# sanitize workspace (so extra files aren't added to the container)


# build the container
echo "sudo docker build -t trt_pose:r$L4T_VERSION -f ${SCRIPTPATH}/Dockerfile \
          --build-arg BASE_IMAGE=$BASE_IMAGE \
		."
sudo docker build -t trt_pose:r$L4T_VERSION -f ${SCRIPTPATH}/Dockerfile \
          --build-arg BASE_IMAGE=$BASE_IMAGE \
	  	.
