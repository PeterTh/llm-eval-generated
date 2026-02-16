cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Define that we are using MPI, OpenMP, and CUDA
target_compile_definitions(cholesky PRIVATE USE_MPI USE_OPENMP USE_CUDA)

# Enable optimizations and ensure nvcc forwards OpenMP flag to host compiler
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Link math, MPI, OpenMP, and CUDA runtime
target_link_libraries(cholesky PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX CUDAToolkit::cudart)
