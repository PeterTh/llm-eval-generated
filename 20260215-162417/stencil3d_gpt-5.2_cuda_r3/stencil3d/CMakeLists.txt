cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ensure we compile for the local GPU for best performance when supported.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  cmake_policy(SET CMP0104 NEW)
  if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()

add_executable(stencil3d stencil3d.cpp)

# Compile the existing TU with NVCC (no file renames).
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
# Note: keep CUDA math precise (no --use_fast_math) to preserve semantics.
target_compile_options(stencil3d PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(stencil3d m)
