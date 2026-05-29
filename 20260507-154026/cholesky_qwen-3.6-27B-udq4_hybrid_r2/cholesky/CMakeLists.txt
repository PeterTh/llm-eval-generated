cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Compile with nvcc for CUDA device code
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Host / device compile options
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Link MPI, OpenMP, cuSOLVER and math
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
        cusolver cublas m)

set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "" FORCE)
message(STATUS "CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
