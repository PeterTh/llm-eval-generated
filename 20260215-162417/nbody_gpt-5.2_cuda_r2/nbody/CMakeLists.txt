cmake_minimum_required(VERSION 3.10)

# nvcc isn't always on PATH in this environment.
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# Compile the existing source as CUDA (no new files).
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Build for the local GPU when available (CMake >= 3.24).
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  set_property(TARGET nbody PROPERTY CUDA_ARCHITECTURES native)
endif()

# Enable optimizations
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(nbody m)
