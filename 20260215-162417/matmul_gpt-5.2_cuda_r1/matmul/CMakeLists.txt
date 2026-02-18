cmake_minimum_required(VERSION 3.10)

# Help CMake find nvcc even when it's not on PATH (common in minimal CI images).
if (NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "" FORCE)
endif()

# Avoid configure-time CUDA arch auto-detection failures on systems without a visible GPU.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
# NOTE: matmul.cpp is compiled as CUDA, so host flags must be passed via -Xcompiler.
target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(matmul m)
