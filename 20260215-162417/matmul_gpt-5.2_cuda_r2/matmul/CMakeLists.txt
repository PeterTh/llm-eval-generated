cmake_minimum_required(VERSION 3.18)

# Set architectures *before* enabling CUDA to avoid "default arch" detection failures.
# Override at configure time via -DCMAKE_CUDA_ARCHITECTURES=...
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Compile matmul.cpp with NVCC (no new files).
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(matmul m)
