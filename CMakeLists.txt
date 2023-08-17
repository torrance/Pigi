cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "hipcc")
set(CMAKE_CXX_STANDARD "20")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")

project(Pigi)

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
find_library(gsl NAMES gsl REQUIRED)
find_library(cfitsio NAMES cfitsio REQUIRED)
find_library(fmt NAMES fmt REQUIRED)
find_library(hipfft NAMES hipfft HINTS ${hippath}/lib REQUIRED)

add_executable(test src/test.cpp)
target_compile_options(test PRIVATE ${hipcxxflags})
target_link_libraries(test PRIVATE ${casa_casa} ${casa_tables} ${gsl} ${cfitsio} ${fmt} ${hipfft} Catch2::Catch2WithMain)