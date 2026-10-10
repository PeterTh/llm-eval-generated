cmake_minimum_required(VERSION 3.24)

# Use the installed GPU by default; callers can select other architectures
# through the standard CMAKE_CUDA_ARCHITECTURES cache variable.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename while compiling it with NVCC.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

# Preserve the reference's floating-point evaluation order and precision.
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(stencil3d m)
