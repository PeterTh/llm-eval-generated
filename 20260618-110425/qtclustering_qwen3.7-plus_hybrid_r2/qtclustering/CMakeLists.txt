cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Set source file language to CUDA
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Pass OpenMP flags to CUDA host compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")

# Link MPI, OpenMP, CUDA, and math library
target_link_libraries(qtclustering MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)
set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
