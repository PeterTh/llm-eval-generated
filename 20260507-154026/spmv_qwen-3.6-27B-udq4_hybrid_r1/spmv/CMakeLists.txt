cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Force nvcc to compile the .cpp file (it contains CUDA code)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA properties
set_target_properties(spmv PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86"
    CUDA_SEPARABLE_COMPILATION OFF
)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link MPI, CUDA runtime, and math library
target_link_libraries(spmv
    MPI::MPI_CXX
    cuda
    cudart
    m
)
