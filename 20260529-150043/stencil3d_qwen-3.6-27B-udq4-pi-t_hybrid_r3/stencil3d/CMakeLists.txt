cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA separate compilation for large kernels
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Source file
set(SOURCE_FILES stencil3d.cpp)

add_executable(stencil3d ${SOURCE_FILES})

# Force the .cpp file to be compiled with nvcc (CUDA compiler)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA properties
set_target_properties(stencil3d PROPERTIES
    CUDA_ARCHITECTURES "86"  # Ampere (RTX 30xx)
)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Link MPI, OpenMP, CUDA runtime, and math
target_link_libraries(stencil3d PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cudart
    m
)

# Include directories
target_include_directories(stencil3d PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/..
)
