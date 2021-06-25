#!/usr/bin/env bash
#
# Start an instance of the trt_pose docker container.
# See below or run this script with -h or --help to see usage options.
#
# This script should be run from the root dir of the trt_pose project:
#
#     $ cd /path/to/your/trt_pose
#     $ docker/run.sh
#

show_help() {
    echo " "
    echo "usage: Starts the Docker container and runs a user-specified command"
    echo " "
    echo "   ./docker/run.sh --container DOCKER_IMAGE"
    echo "                   --volume HOST_DIR:MOUNT_DIR"
    echo "                   --jupyter"
    echo " "
    echo " or "
    echo " "
    echo "   ./docker/run.sh --container DOCKER_IMAGE"
    echo "                   --volume HOST_DIR:MOUNT_DIR"
    echo "                   --run RUN_COMMAND"
    echo " "
    echo "args:"
    echo " "
    echo "   --help                       Show this help text and quit"
    echo " "
    echo "   -c, --container DOCKER_IMAGE Specifies the name of the Docker container"
    echo "                                image to use (default: 'nvidia-l4t-base')"
    echo " "
    echo "   -v, --volume HOST_DIR:MOUNT_DIR Mount a path from the host system into"
    echo "                                   the container.  Should be specified as:"
    echo " "
    echo "                                      -v /my/host/path:/my/container/path"
    echo " "
    echo "                                   (these should be absolute paths)"
    echo " "
    echo "   -j, --jupyter          Start Jupyter Lab server and shows the log"
    echo "                          Shutting down the Jupyter server stops the container"
    echo " "
    echo "   -r, --run RUN_COMMAND  Command to run once the container is started."
    echo "                          Note that this argument must be invoked last,"
    echo "                          as all further arguments will form the command."
    echo "                          If no run command is specified, an interactive"
    echo "                          terminal into the container will be provided."
    echo " "
}

die() {
    printf '%s\n' "$1"
    show_help
    exit 1
}

# find container tag from L4T version
source docker/tag.sh

# paths to some project directories

DOCKER_ROOT="/trt_pose"	# where the project resides inside docker

# generate mount commands
DATA_VOLUME="\
--volume $HOME:$DOCKER_ROOT"

# parse user arguments
USER_VOLUME=""
USER_COMMAND=""

while :; do
    case $1 in
        -h|-\?|--help)
            show_help    # Display a usage synopsis.
            exit
            ;;
        -c|--container)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                CONTAINER_IMAGE=$2
                shift
            else
                die 'ERROR: "--container" requires a non-empty option argument.'
            fi
            ;;
        --container=?*)
            CONTAINER_IMAGE=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --container=)         # Handle the case of an empty --image=
            die 'ERROR: "--container" requires a non-empty option argument.'
            ;;
        -v|--volume)
            if [ "$2" ]; then
                USER_VOLUME=" -v $2 "
                shift
            else
                die 'ERROR: "--volume" requires a non-empty option argument.'
            fi
            ;;
        --volume=?*)
            USER_VOLUME=" -v ${1#*=} " # Delete everything up to "=" and assign the remainder.
            ;;
        --volume=)         # Handle the case of an empty --image=
            die 'ERROR: "--volume" requires a non-empty option argument.'
            ;;
	-j|--jupyter)
	    USER_COMMAND="pwd; jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser"
            ;;
        -r|--run)
            if [ "$2" ]; then
                shift
		if [ -z "$USER_COMMAND" ]; then
                    USER_COMMAND="$@"
		else
                    die 'ERROR: "--run" option cannot be used with "--jupyter" option.'
		fi
            else
                die 'ERROR: "--run" requires a non-empty option argument.'
            fi
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
done

echo "CONTAINER:     $CONTAINER_IMAGE"
echo "DATA_VOLUME:   $DATA_VOLUME"
echo "USER_VOLUME:   $USER_VOLUME"
echo "USER_COMMAND:  $USER_COMMAND"

[ ! -z "$USER_COMMAND" ] && USER_COMMAND="/bin/bash -c \"$USER_COMMAND\"" || echo "USER_COMMAND is empty" 

# check for V4L2 devices
V4L2_DEVICES=" "

for i in {0..9}
do
	if [ -a "/dev/video$i" ]; then
		V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
	fi
done

echo "V4L2_DEVICES:  $V4L2_DEVICES"

# run the container
sudo xhost +si:localuser:root


echo "
sudo docker run --runtime nvidia -it --rm --network host --privileged -e \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /dev/bus/usb:/dev/bus/usb \ 
    $V4L2_DEVICES $DATA_VOLUME $USER_VOLUME \
    $CONTAINER_IMAGE $USER_COMMAND
"

/bin/bash -c "sudo docker run --runtime nvidia -it --rm --network host --privileged \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /dev/bus/usb:/dev/bus/usb \
    $V4L2_DEVICES $DATA_VOLUME $USER_VOLUME \
    $CONTAINER_IMAGE $USER_COMMAND
"
