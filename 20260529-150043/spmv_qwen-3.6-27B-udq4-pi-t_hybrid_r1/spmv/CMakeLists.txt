cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090 (Ampere sm_86)

add_executable(spmv spmv.cpp)

# Compile spmv.cpp with nvcc (CUDA) since it contains device code
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Host compile options: O3, native arch, OpenMP, warnings
target_compile_options(spmv PRIVATE
    -O3 -march=native -Wall -Wextra
    -fopenmp
)

# Link libraries: MPI, CUDA runtime, OpenMP, math
target_link_libraries(spmv PRIVATE
    MPI::MPI_CXX
    ${CUDA_CUDA_LIBRARY}
    ${OpenMP_CXX_LIBRARIES}
    m
)

# Include directories
target_include_directories(spmv PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/..
)
