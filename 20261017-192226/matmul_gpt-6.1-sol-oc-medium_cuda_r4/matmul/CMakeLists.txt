cmake_minimum_required(VERSION 3.24)

# Generate native GPU machine code by default, avoiding first-launch PTX JIT.
# Cross-compiling builds can supply an explicit architecture list instead.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA target architectures")
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA without changing its name.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

# Keep the original ordered double-precision multiply/add rounding. In
# particular, do not enable fast math or reduced-precision tensor operations.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(matmul m)
