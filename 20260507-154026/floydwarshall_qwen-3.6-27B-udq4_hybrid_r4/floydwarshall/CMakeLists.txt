cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile as CUDA for device code support
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Hybrid parallelization flags
# -Xcompiler passes flags to the host compiler (g++) so nvcc handles OpenMP pragmas
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler=-fopenmp)
target_link_libraries(floydwarshall MPI::MPI_CXX cudart m)
target_link_options(floydwarshall PRIVATE -fopenmp)
