cmake_minimum_required(VERSION 3.24)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on the build machine
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(qtclustering qtclustering.cpp)

# The source is CUDA (device kernels), so it is compiled by nvcc
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --compiler-options -march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(qtclustering m)
