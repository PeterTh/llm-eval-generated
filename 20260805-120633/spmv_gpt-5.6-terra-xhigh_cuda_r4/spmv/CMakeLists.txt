cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Keep the existing source filename while compiling the benchmark with nvcc.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# The benchmark environment targets Ampere GPUs.  Compiling native SASS avoids
# a first-run JIT penalty and permits the compiler to optimize for that ISA.
set_target_properties(spmv PROPERTIES CUDA_ARCHITECTURES 86)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(spmv m)
