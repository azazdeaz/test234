# FROM nvidia/cudagl:11.0-base
FROM nvidia/opengl:1.2-glvnd-devel-ubuntu20.04

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && apt-get -qq update && apt-get -q -y install tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq clean

# ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -qq update && apt-get -q -y install \
    gnupg2 \
    curl \
    ca-certificates \
    build-essential \
    git \
    tmux \
    nano \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq clean

# setup sources.list and keys
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# install ROS (including dependencies for building packages) and other packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && apt-get -q -y install \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    ros-noetic-desktop \
    && rosdep init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq clean

# sdformat8-sdf conflicts with sdformat-sdf installed from gazebo
# so we need to workaround this using a force overwrite
# Do this before installing ign-gazebo
# (then install ign-blueprint and ros to ign bridge)
RUN echo "deb [trusted=yes] http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list \
    && wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && apt-get -qq update && apt-get -q -y install \
    ignition-dome \
    ros-noetic-ros-ign \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq clean


RUN apt-get update \
    && apt-get install python3-pip -y\
    && apt-get -qq clean

RUN python3 -m pip install numpy notebook matplotlib tqdm scipy
RUN python3 -m pip install ipywidgets --upgrade
RUN jupyter nbextension enable --py widgetsnbextension




# GTSAM

# Disable GUI prompts
ENV DEBIAN_FRONTEND noninteractive

# Update apps on the base image
RUN apt-get -y update && apt-get -y install

# Install C++
RUN apt-get -y install build-essential  apt-utils

# Install boost and cmake
RUN apt-get -y install libboost-all-dev cmake

# Install TBB
RUN apt-get -y install libtbb-dev

# Install pip
RUN apt-get install -y python3-pip python3-dev

# Install compiler
RUN apt-get install -y build-essential

# Clone GTSAM (develop branch)
WORKDIR /usr/src/
RUN git clone --single-branch --branch develop https://github.com/borglab/gtsam.git

# Install python wrapper requirements
RUN python3 -m pip install -U -r /usr/src/gtsam/python/requirements.txt

# Run cmake again, now with python toolbox on
WORKDIR /usr/src/gtsam/build
RUN cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_WITH_EIGEN_MKL=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING_ALWAYS=OFF \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_PYTHON=ON \
    -DGTSAM_PYTHON_VERSION=3\
    ..

# Build 
RUN make -j6 install
RUN make -j6 python-install
# RUN make clean

# Needed to link with GTSAM
RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib:LD_LIBRARY_PATH' >> /root/.bashrc

# Needed to run python wrapper:
RUN echo 'export PYTHONPATH=/usr/local/python/:$PYTHONPATH' >> /root/.bashrc

# Run bash
CMD ["bash"]




# OpenCV
RUN apt-get update \
    && apt-get install -y \
        python3-opencv \
    && apt-get -qq clean

RUN apt-get update \
    && apt-get install -y \
        curl \
    && apt-get -qq clean
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default nightly
RUN cargo install --git https://github.com/eclipse-zenoh/zenoh.git

COPY . /catkin_ws/src
WORKDIR /catkin_ws

RUN rosdep update
RUN rosdep install --from-paths src --ignore-src -r -y  || echo "There were some errors during rosdep install"
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash && \
    catkin_make

ENV WORLD_DIR=/catkin_ws/src/fields_ignition/generated_examples/tomato_field
# CMD source devel/setup.bash && roslaunch fields_ignition field.launch world_dir:=${WORLD_DIR}
