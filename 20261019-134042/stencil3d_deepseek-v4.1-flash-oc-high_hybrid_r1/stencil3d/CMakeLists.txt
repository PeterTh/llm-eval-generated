cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 52)
    endif()
endif()

project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)

# The implementation contains CUDA kernels, so compile the existing source
# file with the CUDA toolchain while keeping the executable name unchanged.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(stencil3d PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
