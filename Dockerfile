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


# TODO move up
ARG NUM_THREADS=8

# OpenCV
WORKDIR /tmp
RUN set -x && \
    apt-get update && \
    apt-get install -y \
        python3-dev \
        python3-numpy \
        libavcodec-dev libavformat-dev libswscale-dev \
        libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
        libgtk-3-dev \
        libpng-dev libjpeg-dev && \
    apt-get -qq clean && \
    git clone --depth 1 -j8 https://github.com/opencv/opencv.git && \
    mkdir -p build && \
    cd build && \
    cmake ../opencv && \
    make -j${NUM_THREADS} && \
    make install && \
    cd /tmp && \
    rm -rf *

# Rust
RUN apt-get update \
    && apt-get install -y \
        curl \
    && apt-get -qq clean
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default nightly
# RUN cargo install --git https://github.com/eclipse-zenoh/zenoh.git

RUN apt-get update \
    && apt-get install -y \
        libzmq3-dev \
    && apt-get -qq clean

# install dependencies via apt
ENV DEBCONF_NOWARNINGS yes
RUN set -x && \
  apt-get update -y -qq && \
  apt-get upgrade -y -qq --no-install-recommends && \
  : "basic dependencies" && \
  apt-get install -y -qq \
    build-essential \
    pkg-config \
    cmake \
    git \
    wget \
    curl \
    tar \
    unzip && \
  : "g2o dependencies" && \
  apt-get install -y -qq \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libglew-dev && \
  : "OpenCV dependencies" && \
  apt-get install -y -qq \
    libjpeg-dev \
    libpng++-dev \
    libtiff-dev \
    libopenexr-dev \
    libwebp-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev && \
  : "other dependencies" && \
  apt-get install -y -qq \
    libyaml-cpp-dev && \
  : "remove cache" && \
  apt-get autoremove -y -qq && \
  rm -rf /var/lib/apt/lists/*

ENV CPATH=${CMAKE_INSTALL_PREFIX}/include:${CPATH}
ENV C_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include:${CPLUS_INCLUDE_PATH}
ENV LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}
# TODO move up
ARG CMAKE_INSTALL_PREFIX=/usr/local 

# Eigen
ARG EIGEN3_VERSION=3.3.7
WORKDIR /tmp
RUN set -x && \
  wget -q https://gitlab.com/libeigen/eigen/-/archive/${EIGEN3_VERSION}/eigen-${EIGEN3_VERSION}.tar.bz2 && \
  tar xf eigen-${EIGEN3_VERSION}.tar.bz2 && \
  rm -rf eigen-${EIGEN3_VERSION}.tar.bz2 && \
  cd eigen-${EIGEN3_VERSION} && \
  mkdir -p build && \
  cd build && \
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    .. && \
  make -j${NUM_THREADS} && \
  make install && \
  cd /tmp && \
  rm -rf *
ENV Eigen3_DIR=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake



# g2o
ARG G2O_COMMIT=9b41a4ea5ade8e1250b9c1b279f3a9c098811b5a
WORKDIR /tmp
RUN set -x && \
  git clone https://github.com/RainerKuemmerle/g2o.git && \
  cd g2o && \
  git checkout ${G2O_COMMIT} && \
  mkdir -p build && \
  cd build && \
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_UNITTESTS=OFF \
    -DBUILD_WITH_MARCH_NATIVE=OFF \
    -DG2O_USE_CHOLMOD=OFF \
    -DG2O_USE_CSPARSE=ON \
    -DG2O_USE_OPENGL=OFF \
    -DG2O_USE_OPENMP=ON \
    -DG2O_BUILD_APPS=OFF \
    -DG2O_BUILD_EXAMPLES=OFF \
    -DG2O_BUILD_LINKED_APPS=OFF \
    .. && \
  make -j${NUM_THREADS} && \
  make install && \
  cd /tmp && \
  rm -rf *
ENV g2o_DIR=${CMAKE_INSTALL_PREFIX}/lib/cmake/g2o

# DBoW2
ARG DBOW2_COMMIT=687fcb74dd13717c46add667e3fbfa9828a7019f
WORKDIR /tmp
RUN set -x && \
  git clone https://github.com/OpenVSLAM-Community/DBoW2.git && \
  cd DBoW2 && \
  git checkout ${DBOW2_COMMIT} && \
  mkdir -p build && \
  cd build && \
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    .. && \
  make -j${NUM_THREADS} && \
  make install && \
  cd /tmp && \
  rm -rf *
ENV DBoW2_DIR=${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2


# socket.io-client-cpp
ARG SIOCLIENT_COMMIT=ff6ef08e45c594e33aa6bc19ebdd07954914efe0
WORKDIR /tmp
RUN set -x && \
  git clone https://github.com/shinsumicco/socket.io-client-cpp.git && \
  cd socket.io-client-cpp && \
  git checkout ${SIOCLIENT_COMMIT} && \
  git submodule init && \
  git submodule update && \
  mkdir -p build && \
  cd build && \
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    -DBUILD_UNIT_TESTS=OFF \
    .. && \
  make -j${NUM_THREADS} && \
  make install && \
  cd /tmp && \
  rm -rf *
ENV sioclient_DIR=${CMAKE_INSTALL_PREFIX}/lib/cmake/sioclient

# protobuf
WORKDIR /tmp
RUN set -x && \
  apt-get update -y -qq && \
  apt-get upgrade -y -qq --no-install-recommends && \
  apt-get install -y -qq autogen autoconf libtool && \
  wget -q https://github.com/google/protobuf/archive/v3.6.1.tar.gz && \
  tar xf v3.6.1.tar.gz && \
  cd protobuf-3.6.1 && \
  ./autogen.sh && \
  ./configure --prefix=${CMAKE_INSTALL_PREFIX} --enable-static=no && \
  make -j${NUM_THREADS} && \
  make install && \
  cd /tmp && \
  rm -rf * && \
  # apt-get purge -y -qq autogen autoconf libtool && \
  # apt-get autoremove -y -qq && \
  rm -rf /var/lib/apt/lists/*


# COPY . /catkin_ws/src
WORKDIR /catkin_ws
RUN mkdir src

RUN rosdep update
RUN rosdep install --from-paths src --ignore-src -r -y  || echo "There were some errors during rosdep install"
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash && \
    catkin_make_isolated

RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

ENV WORLD_DIR=/catkin_ws/src/fields_ignition/generated_examples/tomato_field
# CMD source devel/setup.bash && roslaunch fields_ignition field.launch world_dir:=${WORLD_DIR}
