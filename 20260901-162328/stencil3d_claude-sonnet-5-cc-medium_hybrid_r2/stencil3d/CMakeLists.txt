cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile the single source file as CUDA so it can contain both host and
# device (kernel) code alongside the MPI/OpenMP host code.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES LINKER_LANGUAGE CUDA)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-Wall,-Wextra,-fopenmp -O3 --use_fast_math>
)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link math, MPI, OpenMP, and CUDA runtime libraries
target_link_libraries(stencil3d PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_link_options(stencil3d PRIVATE -fopenmp)
