cmake_minimum_required(VERSION 3.10)

# nvcc is installed but may not be on PATH in some runners.
if (NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "" FORCE)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the local GPU.
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(roomsim roomsim.cpp)

# Compile this translation unit with nvcc so we can use CUDA kernels.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

find_package(CUDAToolkit REQUIRED)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_link_libraries(roomsim PRIVATE CUDA::cudart m)
