cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(CMAKE_CXX_STANDARD "20")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")

project(Pigi)

# Default build = Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Find hipconfig
find_program(hipconfig hipconfig REQUIRED)

# Get hip compiler flags
execute_process(COMMAND ${hipconfig} --cpp_config OUTPUT_VARIABLE hipcxxflags)
separate_arguments(hipcxxflags UNIX_COMMAND ${hipcxxflags})

# Add hip installation to search directories
execute_process(COMMAND ${hipconfig} --path OUTPUT_VARIABLE hippath)
include_directories(${hippath})

find_package(Catch2 3 REQUIRED)

# Required libraries
find_library(casa_casa NAMES casa_casa REQUIRED)
find_library(casa_tables NAMES casa_tables REQUIRED)
find_path(casa_INCLUDE_DIRS casacore/tables/Tables.h)
find_library(gsl NAMES gsl REQUIRED)
find_path(gsl_INCLUDE_DIR gsl/gsl_blas.h)
find_library(cfitsio NAMES cfitsio REQUIRED)
find_path(cfitsio_INCLUDE_DIR fitsio.h) 
find_library(fmt NAMES fmt REQUIRED)
find_path(fmt_INCLUDE_DIR fmt/format.h)
find_library(hipfft NAMES hipfft HINTS ${hippath}/lib REQUIRED)

add_executable(test src/test.cpp)
target_compile_options(test PRIVATE ${hipcxxflags})
target_include_directories(test PRIVATE ${fmt_INCLUDE_DIR} ${gsl_INCLUDE_DIR} ${thrust_INCLUDE_DIR} ${cfitsio_INCLUDE_DIR} ${casa_INCLUDE_DIRS})
target_link_libraries(test PRIVATE ${casa_casa} ${casa_tables} ${gsl} ${cfitsio} ${fmt} ${hipfft} Catch2::Catch2WithMain)

add_executable(benchmark src/benchmark.cpp)
target_compile_options(benchmark PRIVATE ${hipcxxflags})
target_include_directories(benchmark PRIVATE ${fmt_INCLUDE_DIR} ${gsl_INCLUDE_DIR} ${thrust_INCLUDE_DIR} ${cfitsio_INCLUDE_DIR} ${casa_INCLUDE_DIRS})
target_link_libraries(benchmark PRIVATE ${casa_casa} ${casa_tables} ${gsl} ${cfitsio} ${fmt} ${hipfft} Catch2::Catch2WithMain)
