#!/bin/bash

# Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Script to install HDF5 from source

set -euo pipefail

echo "Installing zlib with yum"
yum -y install zlib-devel

pushd /tmp

echo "Downloading & unpacking HDF5 ${HDF5_VERSION}"
#                                   Remove trailing .*, to get e.g. '1.12' ↓
curl -fsSLO "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.gz"
tar -xzvf hdf5-${HDF5_VERSION}.tar.gz

pushd hdf5-${HDF5_VERSION}
echo "Configuring, building & installing HDF5 ${HDF5_VERSION}"
mkdir build
cd build
# Overriding CMAKE_INSTALL_PREFIX is necessary since HDF5 installs into
# /usr/local/HDF_Group/HDF5/${HDF5_VERSION} by default.
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=Off -DHDF5_BUILD_CPP_LIB=On ..
make -j${THREADS}
make install
popd

# Clean up to limit the size of the Docker image
echo "Cleaning up unnecessary files"
rm -r hdf5-${HDF5_VERSION}
rm hdf5-${HDF5_VERSION}.tar.gz

popd

yum -y erase zlib-devel
