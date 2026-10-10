cmake_minimum_required(VERSION 3.18)
# Target the available GPU unless the caller specifies deployment architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(stencil3d_benchmark LANGUAGES CXX CUDA)

# Keep the existing source filename while compiling its kernels with nvcc.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

# Preserve IEEE double arithmetic; fast math would change stencil results.
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)
target_link_libraries(stencil3d PRIVATE m)
