cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Treat .cpp as CUDA source
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# CUDA architecture
set_target_properties(cholesky PROPERTIES CUDA_ARCHITECTURES "70;80;90")

# Enable optimizations
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-march=native,-fopenmp>)

# MPI
find_package(MPI REQUIRED)
target_link_libraries(cholesky MPI::MPI_CXX)

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(cholesky OpenMP::OpenMP_CXX)

# CUDA libraries
target_link_libraries(cholesky cublas cusolver cudart m)
