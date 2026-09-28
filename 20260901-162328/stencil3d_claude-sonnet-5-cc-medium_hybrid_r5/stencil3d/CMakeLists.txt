cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(stencil3d stencil3d.cpp)

# Compile the single source file as CUDA so host (MPI/OpenMP) and device
# (stencil kernel) code can coexist in one translation unit.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations for both the CUDA device code and the host compiler
# invoked through nvcc.
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_options(stencil3d PRIVATE -fopenmp)

target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX m)
