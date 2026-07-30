cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Use nvcc to compile the CUDA parts via separate compilation
# Since spmv.cpp contains both CUDA device code and host code,
# we compile it with nvcc (which handles both)

add_executable(spmv spmv.cpp)

# Force nvcc to compile spmv.cpp since it contains CUDA device code
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA properties
set_target_properties(spmv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

# Link libraries
target_link_libraries(spmv PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cudart
    cusparse
    m
)

# Include directories
target_include_directories(spmv PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link directories for CUDA
target_link_directories(spmv PRIVATE
    ${CMAKE_CUDA_TOOLKIT_LIB_DIRECTORIES}
)
