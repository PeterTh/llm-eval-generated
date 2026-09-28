cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Compile the source as CUDA so it can contain __global__ kernels while
# keeping the original .cpp file name/extension.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-Wall,-Wextra>
)

target_link_libraries(cholesky m CUDA::cudart)
