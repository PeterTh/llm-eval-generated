cmake_minimum_required(VERSION 3.10)

project(qtclustering CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA is used through the driver API (cuda.h) and NVRTC (runtime kernel
# compilation), so the CUDA *language* is not needed and the source stays
# a plain .cpp compiled by the host compiler. We only need the CUDA
# Toolkit's headers/libraries.
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

target_include_directories(qtclustering PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Link math, MPI, OpenMP, and the CUDA driver + NVRTC libraries.
target_link_libraries(qtclustering
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cuda_driver
    CUDA::nvrtc
)
