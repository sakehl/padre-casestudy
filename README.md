# Case Study Padre
This is the accompanying code for the paper `Verifying a Radio Telescope Pipeline using HaliVer:  Solving Nonlinear and Quantifier Challenges`.

## Installing
We support running the code via the docker image
[`Dockerfile`](Dockerfile). 

Run:
```cmd
docker build -t padre .
```
to build the Dockerfile.

With
```
docker run -it --entrypoint bash padre
```
You can run it.

# DP3
LOFAR preprocessing software, including averaging, flagging, various kinds of calibration and more.

The DP3 documentation can be found at: https://dp3.readthedocs.org

This repository is a continuation of the one at svn.astron.nl/LOFAR. In particular, it has branched off at LOFAR Release 3.2 (Sept 2018). The version of DP3 that is in the ASTRON repository is no longer maintained.

## Installation
Some non-standard dependencies of this project are: armadillo, boost, boost-python, casacore, hdf5, aoflagger, and EveryBeam. See the Dockerfiles [`docker/ubuntu_20_04_base`](docker/ubuntu_20_04_base) and/or [`docker/ubuntu_22_04_base`](docker/ubuntu_22_04_base) as examples.

Typical installation commands:
```
mkdir build
cd build
cmake ..
make -j4
make install
```
## Contributing

### Want to Help?

Issues can be filed either at [gitlab](https://git.astron.nl/RD/DP3) or [github](https://github.com/lofar-astron/DP3).

Want to contribute some code, or improve documentation? You can start by cloning our code from the [DP3 development repository](https://git.astron.nl/RD/DP3).

# Standard build command
```
CUDACXX=/usr/local/cuda-12/bin/nvcc cmake -B build -S . -DBUILD_TESTING=ON -DBUILD_WITH_HALIDE=ON -D BUILD_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
```
```
cmake --build build -- -j8
```

## Testing
```
./build/unittests "-t" "solvers/iterative_diagonal_dd_intervals"
./build/unittests "-t" "solvers/iterative_diagonal_halide"
./build/unittests "-t" "solvers/iterative_diagonal_halide_parts"
./build/unittests "-t" "solvers/iterative_diagonal_halide_time"
```