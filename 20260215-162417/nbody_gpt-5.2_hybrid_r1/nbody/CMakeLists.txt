cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA runtime (no external deps; uses system CUDA install).
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
  HINTS /usr/local/cuda/include /usr/local/cuda-12.6/include /usr/local/cuda-12/include
)
find_library(CUDA_CUDART_LIBRARY cudart
  HINTS /usr/local/cuda/lib64 /usr/local/cuda-12.6/lib64 /usr/local/cuda-12/lib64
)

if(NOT CUDA_INCLUDE_DIR OR NOT CUDA_CUDART_LIBRARY)
  message(FATAL_ERROR "CUDA runtime headers/libraries not found (cuda_runtime.h, libcudart)")
endif()

add_executable(nbody nbody.cpp)

target_include_directories(nbody PRIVATE ${CUDA_INCLUDE_DIR})

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX ${CUDA_CUDART_LIBRARY} m)
