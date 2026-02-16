cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Compile with optimizations, enable OpenMP for host compiler and pass OpenMP flag through nvcc
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Link MPI, OpenMP and math library
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
