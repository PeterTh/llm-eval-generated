cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Ensure the .cpp is compiled with the CUDA compiler so kernels can be in the same file
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and parallel flags
target_compile_options(spmv PRIVATE -O3 -Wall -Wextra)
# Host compiler OpenMP
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
# For NVCC, pass through OpenMP flag to host compiler
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)

# Link libraries: MPI, OpenMP and math
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Set properties for CUDA compilation (reasonable defaults)
if (CMAKE_CUDA_COMPILER)
    set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # Allow modern C++ for NVCC
    set_target_properties(spmv PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()
