cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Use the MPI compiler wrapper as CUDA's host compiler so MPI headers/libs
# are picked up transparently for the mixed MPI+OpenMP+CUDA translation unit.
set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the single source file as CUDA so it can contain both host
# (MPI/OpenMP) code and __global__ device kernels.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations; forward host-side flags to the underlying compiler.
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

set_target_properties(stencil3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(stencil3d PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
