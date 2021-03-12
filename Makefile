build:
	docker build -t py-slam-test .

build-rpi:
	docker build -f Dockerfile.rpi -t py-slam-test-rpi .

run:
	docker run -it \
		--name slam_test \
		-v "/dev/input:/dev/input" \
		--network host \
		--rm \
		--privileged \
		--mount src="`pwd`/notebooks",target=/catkin_ws/src/notebooks,type=bind \
		--mount src="/home/azazdeaz/repos/test/mono-vo/dataset/kitti05/image_0/",target=/images,type=bind,readonly \
		-p 8888:8888 \
		azazdeaz/ps-test \
		bash -c "jupyter notebook /catkin_ws/src/notebooks --allow-root --ip 0.0.0.0 --port 8888 --no-browser"

run-desktop:
	xhost + \
	&& docker run -it \
		--name slam_test \
		-e WORLD_DIR=/catkin_ws/src/fields_ignition/generated_examples/tomato_field \
		-e DISPLAY \
		-e QT_X11_NO_MITSHM=1 \
		-e XAUTHORITY=`XAUTH` \
		-v "`XAUTH`:`XAUTH`" \
		-v "/tmp/.X11-unix:/tmp/.X11-unix" \
		-v "/etc/localtime:/etc/localtime:ro" \
		-v "/dev/input:/dev/input" \
		--network host \
		--rm \
		--privileged \
		--runtime=nvidia \
		--security-opt seccomp=unconfined \
		--mount src="`pwd`/fields_ignition/scripts",target=/catkin_ws/src/fields_ignition/scripts,type=bind \
		--mount src="/home/azazdeaz/repos/test/mono-vo/dataset/kitti05/image_0/",target=/images,type=bind,readonly \
		-p 8888:8888 \
		-p 5555:5555 \
		py-slam-test \
		bash -c "ls && source devel/setup.bash && jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser"