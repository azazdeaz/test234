build:
	docker build -t py-slam-test .

run:
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
		py-slam-test \
		bash -c "ls && source devel/setup.bash && jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser"