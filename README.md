# Pigi

The Parallel Interferometric GPU Imager (Pigi; pronounced like 'piggy') is a prototypal imager for radio astronomy currently in development, with the eventual aim to be optimized to run on supercomputing topology (multiple compute nodes) and take advantage of graphics processing units (GPUs).

It is not (yet?) intended for general use.

This work is funded by the Curtin Institute for Data Science and the Pawsey Supercomputing Centre.

## Installing

HIP is a tricky environment to set up correctly (in addition to all the other dependencies of Pigi), so have made an Apptainer/Singularity definition file available to build the required environment.

Download the [`singuarity.def`](https://github.com/torrance/Pigi/blob/main/singularity.def) build script and create your container:

    singularity build --fakeroot --build-arg platform=nvidia|amd singularity.sif singularity.def

(Substitute either `nvidia` or `amd` for the platform depending on your system.)

Once your container is build you will need to build Pigi. Enter a Singularity shell, being sure to make the GPU available to your session. For NVIDIA:

    singularity shell --nvccli singularity.sif

Or for AMD:

    singularity shell --rocm singularity.sif

Navigate to the Pigi directory and from here build Pigi using the usual CMake incantations:

    mkdir build && cd build
    CMake ..
    make main

## Testing and Benchmarking

### Obtaining test data

To run the tests and benchmarks in full, you need to first download a test data set onto your system.

In a directory of your choosing, run:

    wget https://data.pawsey.org.au/download/mwasci/torrance/1215555160.ms.zip && unzip 1215555160.ms.zip

### Building and running

Next, build the `test` and `benchmark` programs:

    make test
    make benchmark

Finally, run with the `TESTDATA` environment variable set to the location of the test data folder. For example:

    TESTDATA=/my/path/1215555160.ms ./test

Or for benchmarks, similarly:

    TESTDATA=/my/path/1215555160.ms ./benchmark

### Other options

The test and benchmarks use [Catch2](https://github.com/catchorg/Catch2/), and all the standard Catch2 command line options are available. See [here](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#top) for a full description.

Of note, it is possible to run specific tests or benchmarks. You can see a full list of tests and their tags by running:

    ./test --list-tests

From this selection, you can pass in a subset of tests to run. To run just the `invert` benchmark, for example, run:

    TESTDATA=/my/path/1215555160.ms ./benchmark [invert]

## Running

`Pigi` is run by passing in a configuration file in a `TOML` format along with one or more measurement sets:

    mpirun -n 2 pigi --config path/to/config.toml data1.ms [data2.ms, ...]

`Pigi` is _always_ run using `mpi`, and requires `(n + 1)` processes, where `n` is the `channels-out` parameter.

A template configuration file can be created using the command:

    pigi --makeconfig > config.toml

You will need to change some of the default values in this file, especially the phase and projection centers.

### Singularity

If you have installed `Pigi` using a singularity container, you can use the `exec` command. For example, for NVIDIA:

    singularity exec --nvccli singularity.sif mpirun -n path/to/pigi --config path/to/config.toml data1.ms [data2.ms, ...]