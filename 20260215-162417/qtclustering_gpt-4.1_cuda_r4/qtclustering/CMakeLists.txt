cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "-O3 -arch=sm_60 -std=c++17")

find_package(CUDA REQUIRED)

cuda_add_executable(qtclustering qtclustering.cpp)

# Link math library
cuda_add_cublas_to_target(qtclustering)
target_link_libraries(qtclustering m)
