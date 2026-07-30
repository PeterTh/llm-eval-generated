cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

# Set CUDA architecture for performance
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_STANDARD 17)

add_executable(black_scholes black_scholes.cu)

# Enable optimizations for host code
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

# Link MPI, OpenMP, CUDA runtime, and math libraries
target_link_libraries(black_scholes
    PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${CUDA_CUDART_LIBRARY}
    m
)

# Include directories
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/..)
