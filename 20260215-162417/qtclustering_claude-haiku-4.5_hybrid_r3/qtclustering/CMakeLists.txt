cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Enable hybrid parallelization
target_compile_definitions(qtclustering PRIVATE 
    USE_MPI 
    USE_CUDA
)

# Compiler optimizations
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link required libraries
target_link_libraries(qtclustering 
    PRIVATE
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
)

# Include CUDA directories
target_include_directories(qtclustering PRIVATE ${CUDA_INCLUDE_DIRS})
