FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# TODO: needs to be bumped before next DP3 release
# ENV IDG_VERSION=0.8
ENV EVERYBEAM_VERSION=v0.5.1
ENV IDG_VERSION=6b61c038883ad3f807d20047c4f9e1a1f0b8d98a
ENV AOFLAGGER_VERSION=65d5fba4f4c12797386d3fd9cd76734956a8b233

RUN export DEBIAN_FRONTEND="noninteractive" && \
	apt-get update && \
    apt-get install -y \
		bison \
		build-essential \
		casacore-dev \
		casacore-tools \
		clang-format-14 \
		cmake \
		doxygen \
		flex \
		gcovr \
		gfortran \
		git \
		libarmadillo-dev \
		libboost-date-time-dev \
		libboost-filesystem-dev \
		libboost-program-options-dev \
		libboost-python-dev \
		libboost-system-dev \
		libboost-test-dev \
		libcfitsio-dev \
		libfftw3-dev \
		libgsl-dev \
		libgtkmm-3.0-dev \
		libhdf5-serial-dev \
		liblua5.3-dev \
		libpng-dev \
		ninja-build \
		pkg-config \
		pybind11-dev \
		python3-dev \
		python3-numpy \
		python3-pip \
		wcslib-dev \
		wget \
	&& \
	rm -rf /var/lib/apt/lists/*
# Build aoflagger3
RUN mkdir /aoflagger && cd /aoflagger \
    && git clone https://gitlab.com/aroffringa/aoflagger.git src \
    && ( cd src/ && git checkout ${AOFLAGGER_VERSION} ) \
    && mkdir build && cd build \
    && cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr ../src \
    && ninja install \
    && cd / && rm -rf aoflagger
# Build IDG
# (PORTABLE: it may run on a different node than where it was compiled)
RUN mkdir /idg && cd /idg \
    && git clone https://git.astron.nl/RD/idg.git src \
    && ( cd src/ && git checkout ${IDG_VERSION} ) \
    && mkdir build && cd build \
    && cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr -DPORTABLE=ON ../src \
    && ninja install \
    && cd / && rm -rf idg
# Build EveryBeam
RUN mkdir /everybeam && cd /everybeam \
    && git clone https://git.astron.nl/RD/EveryBeam.git src \
    && ( cd src/ && git checkout ${EVERYBEAM_VERSION} ) \
    && mkdir build && cd build \
    && cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr ../src -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    && ninja install \
    && cd / && rm -rf everybeam
# Install WSRT Measures (extra casacore data, for integration tests)
# Note: The file on the ftp site is updated daily. When warnings regarding leap
# seconds appear, ignore them or regenerate the docker image.
RUN wget -nv -O /WSRT_Measures.ztar ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar \
    && cd /var/lib/casacore/data \
    && tar xfz /WSRT_Measures.ztar \
    && rm /WSRT_Measures.ztar
# Install pip dependencies
RUN pip3 install \
		black \
		cmake-format \
		h5py \
		pytest \
		sphinx \
		sphinx-rtd-theme \
	;

RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list

RUN apt update && apt-get install -y --no-install-recommends \
           ca-certificates build-essential curl git wget unzip \
           cmake ninja-build zlib1g-dev \
           clang-11 llvm-11-dev libclang-11-dev liblld-11 liblld-11-dev \
           clang-17 llvm-17-dev libclang-17-dev liblld-17 liblld-17-dev \
           openjdk-17-jre-headless

ENV VERCORS_VERSION=4c01568ea2d55fe149a8b0d253b15e54c3b70bfb

RUN git clone -b 'small-fix' --single-branch --depth 1 https://github.com/sakehl/vercors.git /vercors \
    && cd /vercors && git checkout ${VERCORS_VERSION}

RUN cd /vercors && bin/vct --version

ENV HALIDE_VERSION=a132246ced07adc59c7b3631009464e5a14e0abb

RUN mkdir /halide && cd /halide \
    && git clone https://github.com/halide/Halide.git /halide \
    && (git checkout ${HALIDE_VERSION} ) \
    && mkdir build && \
    cmake -G Ninja \
    -DWITH_TESTS=NO -DWITH_AUTOSCHEDULERS=NO -DWITH_PYTHON_BINDINGS=NO -DWITH_TUTORIALS=NO -DWITH_DOCS=NO -DCMAKE_BUILD_TYPE=Release \
    -DTARGET_AARCH64=NO -DTARGET_AMDGPU=NO -DTARGET_ARM=NO -DTARGET_HEXAGON=NO -DTARGET_MIPS=NO -DTARGET_POWERPC=NO \
    -DTARGET_RISCV=NO -DTARGET_WEBASSEMBLY=NO \
    -S . -B build && cmake --build build

RUN mkdir /halide/install && cmake --install /halide/build --prefix /halide/install


ENV HALIVER_VERSION=f32f2840fc31cf0ef7aee8306dc75d823bfe7cee

RUN mkdir /haliver && cd /haliver \
    && git clone -b 'annotated_halide' --single-branch --depth 1 https://github.com/sakehl/halide.git /haliver \
    && (git checkout ${HALIVER_VERSION} )

RUN cd /haliver && mkdir build && \
    cmake -G Ninja \
    -DWITH_TESTS=NO -DWITH_AUTOSCHEDULERS=NO -DWITH_PYTHON_BINDINGS=NO -DWITH_TUTORIALS=NO -DWITH_DOCS=NO -DCMAKE_BUILD_TYPE=Release \
    -DTARGET_AARCH64=NO -DTARGET_AMDGPU=NO -DTARGET_ARM=NO -DTARGET_HEXAGON=NO -DTARGET_MIPS=NO -DTARGET_NVPTX=NO -DTARGET_POWERPC=NO \
    -DTARGET_RISCV=NO -DTARGET_WEBASSEMBLY=NO \
    -DHalide_REQUIRE_LLVM_VERSION=11.1.0 -Wno-dev -DLLVM_PACKAGE_VERSION=11.1.0 -DLLVM_DIR=/usr/lib/llvm-11/lib/cmake/llvm \
    -S . -B build && cmake --build build

RUN mkdir /haliver/install && cmake --install /haliver/build --prefix /haliver/install

WORKDIR /padre

ENV PADRE_VERSION=5d57e660c5b26c25a3a97b1f999533abbae3fd45

RUN git clone -b 'HaliVer_V1' --single-branch --depth 1 https://github.com/sakehl/padre-casestudy.git /padre \
    && (git checkout ${PADRE_VERSION} ) \
    && git submodule update --init --recursive

RUN cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_WITH_HALIDE=ON -DBUILD_WITH_HALIVER=OFF \
  -DBUILD_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DBUILD_HALIDE_LLVM=OF -DHalide_ROOT=/halide/install \
   -B build -S . 

RUN cmake --build build -- -j8

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/compat