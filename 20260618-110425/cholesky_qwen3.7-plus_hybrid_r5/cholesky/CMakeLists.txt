cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Include directories
target_include_directories(cholesky PRIVATE 
    ${MPI_CXX_INCLUDE_DIRS}
    ${OpenMP_CXX_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cholesky 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    m
)

# Set CUDA properties
set_target_properties(cholesky PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
