# Pigi

The Parallel Interferometric GPU Imager (Pigi; pronounced like 'piggy') is a prototypal imager for radio astronomy currently in development, with the eventual aim to be optimized to run on supercomputing topology (multiple compute nodes) and take advantage of graphics processing units (GPUs).

It is not (yet?) intended for general use.

This work is funded by the Curtin Institute for Data Science and the Pawsey Supercomputing Centre.

## Installing

### Recommended: Using a container

HIP is a tricky environment to set up correctly (in addition to all the other dependencies of Pigi), so we have made an Apptainer/Singularity definition file available to build the required environment and Pigi itself.

Download the [`singuarity.def`](https://github.com/torrance/Pigi/blob/main/singularity.def) build script and create your container:

    # NVIDIA
    singularity build --fakeroot --build-arg platform=nvidia pig.sif --build-arg arch=??? singularity.def

    # Or: AMD
    singularity build --fakeroot --build-arg platform=amd pigi.sif --build-arg arch=??? singularity.def

Note that you will need to provide the GPU architecture for your device as the `arch` argument.

* For NVIDIA, this argument will be passed to `nvcc`'s [`--arch` argument](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch) and will need to be a valid [virtual or real architecture](https://developer.nvidia.com/cuda-gpus). e.g. `--build-arg arch=sm_80` for an A100.
* For AMD  this option is passed to `hipcc`'s `--offload-arch` and will need to be a valid [processor name](https://llvm.org/docs/AMDGPUUsage.html#processors). e.g. `--build-arg arch=gfx90a` for an Instinct MI250.

### Advanced: Using CMake

Pigi uses a standard CMake build process that can be built using the normal series of incantatations:

    cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DGPUARCH=??? ..
    make -j
    make install

However, there are a number of dependencies that must be satisfied first, including:

    * CUDA or ROCM
    * HIP
    * fmt (v10.x)
    * Casacore (>= v3.7)
    * mwa-hyperbeam
    * Boost
    * GSL
    * HDF5
    * CFITSIO
    * WCSLib
    * Catch 2 (for testing)

This list is not exhaustive.

## Testing

### Obtaining test data

To run the tests and benchmarks in full, you need to first download a test data set onto your system.

In a directory of your choosing, run:

    wget -o 1215555160.ms.zip "https://curtin-my.sharepoint.com/:u:/g/personal/277966k_curtin_edu_au/Efh8e_q1IG5DkKCLDl9GNIQBmFS3bM22NF_1B6eLgo3YPQ?download=1" && unzip 1215555160.ms.zip

### Building and running

Run `pigi-test` with the `TESTDATA` environment variable set to the location of the test data folder. For example:

    TESTDATA=/my/path/1215555160.ms pigi-test

Or, using Singularity on NVIDIA:

    TESTDATA=/my/path/1215555160.ms singularity exec --nvccli pigi.sif pigi-test

For AMD, use the `--rocm` flag instead of `--nvccli`.

### Other options

The test and benchmarks use [Catch2](https://github.com/catchorg/Catch2/), and all the standard Catch2 command line options are available. See [here](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#top) for a full description.

It is possible to run specific tests. You can see a full list of tests and their tags by running:

    pigi-test --list-tests

From this selection, you can pass in a subset of tests to run. To run just the `invert` test suite, for example, run:

    TESTDATA=/my/path/1215555160.ms pigi-test [invert]

## Running

`Pigi` is run by passing in a configuration file in a `TOML` format along with one or more measurement sets:

    mpirun -n 2 pigi --config path/to/config.toml data1.ms [data2.ms, ...]

`Pigi` is _always_ run using `mpi`, and requires `(n + 1)` processes, where `n` is the `channels-out` parameter.

A template configuration file can be created using the command:

    pigi --makeconfig > config.toml

You will need to change some of the default values in this file, especially the phase and projection centers.

### Singularity

If you have installed `Pigi` using a singularity container, you can use the `exec` command. For example, for NVIDIA:

    singularity exec --nvccli pigi.sif mpirun -n pigi --config path/to/config.toml data1.ms [data2.ms, ...]

For AMD, use the `--rocm` flag instead of `--nvccli`.