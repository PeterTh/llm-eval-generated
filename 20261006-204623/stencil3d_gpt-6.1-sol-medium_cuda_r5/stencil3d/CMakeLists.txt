cmake_minimum_required(VERSION 3.18)
# CUDA 12.0 does not support GCC 13, which can be the system default.
# Respect an explicit host compiler; otherwise prefer a supported GCC.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(STENCIL3D_CUDA_HOST_COMPILER NAMES g++-12 g++-11 g++)
    if(STENCIL3D_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${STENCIL3D_CUDA_HOST_COMPILER}")
    endif()
endif()

# Let callers choose architectures. On recent CMake, optimize for the GPUs
# on the build machine instead of nvcc's older default architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS}
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA without changing its filename.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)

# CMake links the CUDA runtime automatically for a CUDA executable.
target_link_libraries(stencil3d PRIVATE m)
