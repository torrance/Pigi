# Pigi

The Parallel Interferometric GPU Imager (Pigi; pronounced like 'piggy') is a prototypal imager for radio astronomy currently in development, with the eventual aim to be optimized to run on supercomputing topology (multiple compute nodes) and take advantage of graphics processing units (GPUs).

It is not (yet?) intended for general use.

This work is funded by the Curtin Institute for Data Science and the Pawsey Supercomputing Centre.

## Installing

Pigi uses CMake to build, and so the usual CMake incantation is required:

    mkdir build && cd build
    CMake ..
    make

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