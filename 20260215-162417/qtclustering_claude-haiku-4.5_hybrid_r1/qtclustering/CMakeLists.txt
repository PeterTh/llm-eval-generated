cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages for hybrid parallelization
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Create executable with hybrid parallelization
add_executable(qtclustering qtclustering.cpp)

# Compiler flags for optimization
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries for hybrid parallelization
target_link_libraries(qtclustering 
    PUBLIC MPI::MPI_CXX
    PUBLIC OpenMP::OpenMP_CXX
    PRIVATE m
)

