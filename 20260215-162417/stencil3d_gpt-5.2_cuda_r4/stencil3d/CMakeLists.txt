cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Compile this TU with NVCC so we can keep a single existing source file.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Build for the local GPU for best performance when supported by CMake.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES native)
endif()

# Enable optimizations
target_compile_options(stencil3d PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(stencil3d m)
