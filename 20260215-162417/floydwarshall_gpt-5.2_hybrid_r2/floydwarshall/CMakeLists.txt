cmake_minimum_required(VERSION 3.10)

# CMake may fail to auto-detect a default CUDA architecture in some environments.
# Set a reasonable default for modern accelerator clusters (override via -DCMAKE_CUDA_ARCHITECTURES=... as needed).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# NVCC is not always on PATH in benchmark environments.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Compile this translation unit with NVCC (hybrid MPI+OpenMP+CUDA is unconditional).
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xcompiler=-march=native>
)

target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Prefer static scheduling for OpenMP loops by default (can still be overridden by env).
target_compile_definitions(floydwarshall PRIVATE OMP_SCHEDULE_STATIC=1)
