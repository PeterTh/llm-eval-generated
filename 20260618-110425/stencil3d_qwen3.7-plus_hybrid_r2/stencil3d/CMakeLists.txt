cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(stencil3d stencil3d.cpp)

# Set the source file to be compiled as CUDA
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architecture (compute capability)
set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Enable optimizations
target_compile_options(stencil3d PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xcompiler=-fopenmp,-march=native,-Wall,-Wextra>
)

# Link libraries
target_link_libraries(stencil3d 
    PRIVATE 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

# Include directories
target_include_directories(stencil3d PRIVATE ${CMAKE_SOURCE_DIR}/../common)
