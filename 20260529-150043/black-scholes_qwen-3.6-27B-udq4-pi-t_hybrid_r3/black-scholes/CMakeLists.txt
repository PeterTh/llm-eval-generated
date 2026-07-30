cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

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

# Determine CUDA architecture from available GPUs
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE CUDA_COMPUTE_CAP
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(CUDA_COMPUTE_CAP)
    # Use the first GPU's compute capability (e.g. "8.6" -> "86")
    string(REPLACE "." "" CUDA_COMPUTE_CAP "${CUDA_COMPUTE_CAP}")
    string(REPLACE "\n" " " CUDA_COMPUTE_CAP "${CUDA_COMPUTE_CAP}")
    string(STRIP "${CUDA_COMPUTE_CAP}" CUDA_COMPUTE_CAP)
    # Split into list and take first
    string(REPLACE " " ";" CC_LIST "${CUDA_COMPUTE_CAP}")
    list(GET CC_LIST 0 CC_FIRST)
    set(CMAKE_CUDA_ARCHITECTURES ${CC_FIRST})
    message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    # Fallback: common architectures
    set(CMAKE_CUDA_ARCHITECTURES 86)
    message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES} (fallback)")
endif()

add_executable(black_scholes black_scholes.cpp)

# Tell CMake to compile the .cpp file with nvcc (it contains CUDA device code)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -Xcompiler=-march=native -Wall -Wextra)
target_compile_options(black_scholes PRIVATE ${OpenMP_CXX_FLAGS})

# Link MPI, OpenMP, CUDA runtime, and math
target_link_libraries(black_scholes
    PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cuda
    cudart
    m
)

# Include directories for MPI
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_PATH})
