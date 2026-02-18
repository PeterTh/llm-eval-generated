cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link required libraries
target_link_libraries(spmv
    PRIVATE MPI::MPI_CXX
    PRIVATE OpenMP::OpenMP_CXX
    PRIVATE m
)
