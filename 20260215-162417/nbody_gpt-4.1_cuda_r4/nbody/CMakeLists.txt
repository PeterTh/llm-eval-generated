cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

cuda_add_executable(nbody nbody.cpp)

# Enable optimizations
set(CUDA_NVCC_FLAGS "-O3 -arch=sm_60 -Xcompiler='-Wall,-Wextra'")

# Link math library
target_link_libraries(nbody m)
