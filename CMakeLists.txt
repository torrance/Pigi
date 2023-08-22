cmake_minimum_required(VERSION 3.10)

find_program(CMAKE_CXX_COMPILER "hipcc" HINTS "/opt/rocm/bin")
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
find_library(Lcasa_casa NAMES casa_casa REQUIRED)
find_library(Lcasa_tables NAMES casa_tables REQUIRED)
find_path(Icasa casacore/tables/Tables.h)
find_library(Lgsl NAMES gsl REQUIRED)
find_path(Igsl gsl/gsl_blas.h)
find_library(Lcfitsio NAMES cfitsio REQUIRED)
find_path(Icfitsio_INCLUDE_DIR fitsio.h)
find_library(Lfmt NAMES fmt REQUIRED)
find_path(Ifmt fmt/format.h)
find_library(Lhipfft NAMES hipfft HINTS ${hippath}/lib REQUIRED)

add_executable(test src/test.cpp)
target_compile_options(test PRIVATE ${hipcxxflags})
target_include_directories(test PRIVATE ${Ifmt} ${Igsl} ${Icfitsio} ${Icasa})
target_link_libraries(test PRIVATE ${Lcasa_casa} ${Lcasa_tables} ${Lgsl} ${Lcfitsio} ${Lfmt} ${Lhipfft} Catch2::Catch2WithMain)

add_executable(benchmark src/benchmark.cpp)
target_compile_options(benchmark PRIVATE ${hipcxxflags})
target_include_directories(benchmark PRIVATE ${Ifmt} ${Igsl} ${Icfitsio} ${Icasa})
target_link_libraries(benchmark PRIVATE ${Lcasa_casa} ${Lcasa_tables} ${Lgsl} ${Lcfitsio} ${Lfmt} ${Lhipfft} Catch2::Catch2WithMain)
