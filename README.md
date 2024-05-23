# Case Study Padre
This is the accompanying code for the paper `Verifying a Radio Telescope Pipeline using HaliVer:  Solving Nonlinear and Quantifier Challenges`.

## Installing
We support running the code via the docker image [`Dockerfile`](Dockerfile). 

### Prerequisites
* Have [Docker](https://www.docker.com/) installed.
* Have an Nvidia GPU in your computer.
* Have [CUDA driver 12.4.1](https://developer.nvidia.com/cuda-12-4-1-download-archive) installed.
* Have [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) installed, together with all the [installations steps](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) completed.
* Before proceeding, restart your computer.


### Building docker image
Make sure you have cloned this git repostitory, together with all the submodules.
E.g.
```cmd
git clone https://github.com/sakehl/padre-casestudy.git
cd padre-casestudy
git submodule update --init --recursive
```

Run:
```cmd
docker build -t padre .
```
to build the Dockerfile.

## Running the docker image
Run
```cmd
docker run --rm --runtime=nvidia --gpus all -it --entrypoint bash padre
```
to run the docker image, and open a bash from inside.

## Running the benchmarks
Inside the docker you can run the following commands for the benchmarks
With
```
# Run the regular CPU version
./build/unittests "-t" "solvers/iterative_diagonal_time"
# Run the regular GPU version
./build/unittests "-t" "solvers/iterative_diagonal_cuda_time"
# Run the Halide CPU version
./build/unittests "-t" "solvers/iterative_diagonal_halide_time"
# Run the Halidee GPU version
./build/unittests "-t" "solvers/iterative_diagonal_halide_gpu_time"
```

Note: Your times my vary, as it is dependent on your GPU and CPU from your machine.


## Verification
Run all the following commands from within the docker.

### Verification tries of DDCAL algorithm
All the files we tried to verify can be found in `verification_goals/Experiments` with the folders:
`SolveDirection` `Step` `SubDirection` and `PerformIterationHalide` with different versions as found in the paper.

To verify a file with VerCors you can do:
```cmd
vct FILE
```
E.g. for the file `verification_goals/Experiments/Step/StepHalide-CB.c`
```cmd
vct verification_goals/Experiments/Step/StepHalide-CB.c
```

### Lemmas for nonlinear integer arithmetic  
Inspect the file `verification_goals/lemmas.pvl` for all the lemmas we define and discuss in the paper.
The file can be succesfully verified with
```cmd
vct verification_goals/lemmas.pvl 
```

### Micro benchmarks
The files for the micro benchmarks can be found in `verification_goals/Quant`. Easiest is opening the jupyter notebook there, and execute the tests from there. But in that case you should have installed [VerCors](https://github.com/utwente-fmt/vercors) yourself.
Otherwise you can run the generates files, like 
```cmd
# Normal
vct verification_goals/Quant/data/quant6.pvl
# Predicate
vct verification_goals/Quant/data/quant_p6.pvl
# Sequence
vct verification_goals/Quant/data/quant_s6.pvl
```