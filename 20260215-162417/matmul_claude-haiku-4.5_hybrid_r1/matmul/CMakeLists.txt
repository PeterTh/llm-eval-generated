cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find CUDA compiler
find_program(CUDA_COMPILER_BIN NAMES nvcc)
if(CUDA_COMPILER_BIN)
    message(STATUS "CUDA compiler found at ${CUDA_COMPILER_BIN}, enabling GPU support")
    set(HAVE_CUDA TRUE)
    find_package(CUDA QUIET)
else()
    message(STATUS "CUDA compiler (nvcc) not found, building without GPU support")
    set(HAVE_CUDA FALSE)
endif()

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# MPI configuration
target_link_libraries(matmul PRIVATE MPI::MPI_CXX)

# OpenMP configuration
target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX)

# CUDA configuration (optional)
if(HAVE_CUDA)
    target_compile_definitions(matmul PRIVATE ENABLE_CUDA)
    target_link_libraries(matmul PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(matmul PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

# Math library
target_link_libraries(matmul PRIVATE m)
