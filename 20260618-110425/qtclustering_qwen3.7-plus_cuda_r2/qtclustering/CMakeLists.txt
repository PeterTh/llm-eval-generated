cmake_minimum_required(VERSION 3.10)

project(qtclustering CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (RTX 3090 is compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(qtclustering qtclustering.cu)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link CUDA runtime and math libraries
target_link_libraries(qtclustering cudart m)
