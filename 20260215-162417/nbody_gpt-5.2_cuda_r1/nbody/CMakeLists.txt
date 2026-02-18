cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX)

# CMake 3.30+ requires an explicit architecture when no GPU is detectable.
# Set this before enabling CUDA so compiler detection doesn't fail.
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# nvcc isn't always on PATH in this environment.
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NVCC is more broadly compatible with C++17 device compilation.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# Compile the existing .cpp with NVCC (no file renames).
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

# Link math library
target_link_libraries(nbody m)
