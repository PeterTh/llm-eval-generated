cmake_minimum_required(VERSION 3.18)

# CUDA 12.0 officially supports host GCC <= 12; pick g++-12 if the default is newer
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(GXX12 g++-12)
    if(GXX12)
        set(CMAKE_CUDA_HOST_COMPILER ${GXX12})
    endif()
endif()

project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The stencil source contains CUDA kernels; compile it with nvcc
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations (host-side flags forwarded through nvcc)
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link MPI, OpenMP, and math library
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
