cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Cover common data-center / workstation accelerator generations
    # (Volta, Turing, Ampere, Ada, Hopper)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile the single source file as CUDA so the hybrid MPI/OpenMP/CUDA code builds
# with nvcc while keeping the original file name and executable target.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

target_compile_options(stencil3d PRIVATE
    -O3
    -Xcompiler=-O3,-Wall,-Wextra,-fopenmp
)

target_link_options(stencil3d PRIVATE -fopenmp)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX m)
