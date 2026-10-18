cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA without changing its name.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(stencil3d m)
