cmake_minimum_required(VERSION 3.16)

find_program(CMAKE_CXX_COMPILER "hipcc" PATHS "/opt/rocm/bin")

set(CMAKE_CXX_STANDARD "20")
set(CMAKE_CXX_EXTENSIONS OFF)

# Find hipconfig
find_program(hipconfig hipconfig REQUIRED)

# Determine GPU platform
execute_process(COMMAND ${hipconfig} -P OUTPUT_VARIABLE platform)

# Get hip compiler flags
execute_process(COMMAND ${hipconfig} --cpp_config OUTPUT_VARIABLE hipcxxflags)

# Don't use -Wpedantic for CUDA compilation; causes 'style of line' warnings.
# TODO: add -Wconversion ?
set(warnings "-Wall -Wextra -Wundef -Wuninitialized -Winit-self -Wcast-qual -Wswitch-default -Wswitch-enum -Wunreachable-code")
if ({$platform} STREQUAL amd)
    set(warnings "${warnings} -Wpedantic")
endif()

# Set (platform dependent) CXX compiler flags
if(${platform} STREQUAL amd)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${warnings}")
elseif(${platform} STREQUAL nvidia)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --extended-lambda --expt-relaxed-constexpr -arch=native -Xcompiler \"${warnings}\"")
    set(hipcxxflags "${hipcxxflags} --x cu")
    set(CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Xlinker=-rpath,")
endif()

separate_arguments(hipcxxflags UNIX_COMMAND ${hipcxxflags})

project(Pigi)

# Default build = Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Add hip installation to search directories
execute_process(COMMAND ${hipconfig} --path OUTPUT_VARIABLE hippath)
include_directories(${hippath})

find_package(Catch2 3 REQUIRED)
find_package(MPI REQUIRED COMPONENTS C)
find_package(Boost REQUIRED COMPONENTS mpi program_options serialization)

# Required libraries
find_library(Lcasa_casa NAMES casa_casa REQUIRED)
find_library(Lcasa_tables NAMES casa_tables REQUIRED)
find_library(Lcasa_measures NAMES casa_measures REQUIRED)
find_library(Lcasa_ms NAMES casa_ms REQUIRED)
find_path(Icasa casacore/tables/Tables.h REQUIRED)
find_library(Lgsl NAMES gsl REQUIRED)
find_path(Igsl gsl/gsl_blas.h)
find_library(Lcfitsio NAMES cfitsio REQUIRED)
find_path(Icfitsio fitsio.h REQUIRED)
find_library(Lwcslib NAMES wcs REQUIRED)
find_path(Iwcslib wcslib/wcslib.h REQUIRED)
find_library(Lfmt NAMES fmt REQUIRED)
find_path(Ifmt fmt/format.h REQUIRED)
find_library(Lhipfft NAMES hipfft PATHS ${hippath}/lib REQUIRED)
find_path(Ihyperbeam NAMES mwa_hyperbeam.h REQUIRED)
find_library(Lhyperbeam NAMES mwa_hyperbeam REQUIRED)
find_path(Itoml11 NAMES toml11/toml.hpp PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external REQUIRED)
find_path(Igenerator NAMES generator PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/generator/include REQUIRED)

if(${platform} STREQUAL nvidia)
    find_library(Lcufft NAMES cufft PATHS /usr/local/cuda/lib64 REQUIRED)
endif()

add_executable(main src/main.cpp)
target_compile_options(main PRIVATE ${hipcxxflags})
target_include_directories(main PRIVATE ${Igenerator} ${Ifmt} ${Igsl} ${Icfitsio} ${Iwcslib} ${Icasa} ${Ihyperbeam} ${MPI_C_INCLUDE_DIRS} ${Itoml11})
target_link_libraries(main PRIVATE pthread ${Lcasa_casa} ${Lcasa_tables} ${Lcasa_measures} ${Lcasa_ms} ${Lgsl} ${Lcfitsio} ${Lwcslib} ${Lfmt} ${Lhipfft} ${Lhyperbeam} ${Lcufft} ${MPI_C_LIBRARIES} Boost::mpi Boost::program_options Boost::serialization)

add_executable(test src/test.cpp)
target_compile_options(test PRIVATE ${hipcxxflags})
target_include_directories(test PRIVATE ${Igenerator} ${Ifmt} ${Igsl} ${Icfitsio} ${Iwcslib} ${Icasa} ${Ihyperbeam} ${Itoml11})
target_link_libraries(test PRIVATE pthread ${Lcasa_casa} ${Lcasa_tables} ${Lcasa_measures} ${Lcasa_ms} ${Lgsl} ${Lcfitsio} ${Lwcslib} ${Lfmt} ${Lhipfft} ${Lhyperbeam} ${Lcufft} Catch2::Catch2WithMain)

add_executable(benchmark src/benchmark.cpp)
target_compile_options(benchmark PRIVATE ${hipcxxflags})
target_include_directories(benchmark PRIVATE ${Igenerator} ${Ifmt} ${Igsl} ${Icfitsio} ${Iwcslib} ${Icasa} ${Ihyperbeam})
target_link_libraries(benchmark PRIVATE pthread ${Lcasa_casa} ${Lcasa_tables} ${Lcasa_measures} ${Lcasa_ms} ${Lgsl} ${Lcfitsio} ${Lwcslib} ${Lfmt} ${Lhipfft} ${Lhyperbeam} ${Lcufft} Catch2::Catch2WithMain)
