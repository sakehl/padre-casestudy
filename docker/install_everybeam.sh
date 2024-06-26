#!/bin/bash

# Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Script to install EveryBeam from source

set -euo pipefail

yum -y install wget

pushd /tmp

echo "Cloning EveryBeam ${EVERYBEAM_VERSION}"

git clone --branch v${EVERYBEAM_VERSION} --depth 1 \
  https://git.astron.nl/RD/EveryBeam.git --recursive --shallow-submodules

echo "Configuring, building & installing EveryBeam ${EVERYBEAM_VERSION}"
pushd EveryBeam

# Ensure EveryBeam does not use python.
sed -i '/find_package(PythonInterp .*)/d' CMakeLists.txt
sed -i '/pybind11/d' CMakeLists.txt

mkdir build
cd build
cmake ..
make -j${THREADS} install
popd

# Clean up to limit the size of the Docker image
echo "Cleaning up unnecessary EveryBeam files"
rm -r EveryBeam

popd
