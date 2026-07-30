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

add_executable(stencil3d stencil3d.cpp)

# Set source file language to CUDA
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math --expt-relaxed-constexpr>
)

# Include directories
target_include_directories(stencil3d PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${OpenMP_CXX_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(stencil3d
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

# CUDA architecture (adjust for your GPU)
set_target_properties(stencil3d PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)
