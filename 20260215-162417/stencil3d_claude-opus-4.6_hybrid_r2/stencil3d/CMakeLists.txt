cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Treat .cpp as CUDA source
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# CUDA architecture for RTX 3090 (sm_86), plus common ones
set_target_properties(stencil3d PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86"
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp,-march=native>
)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(stencil3d m ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
