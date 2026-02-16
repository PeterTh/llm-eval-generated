cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Build the single source with NVCC (treat as CUDA)
set(SRC nbody.cpp)
set_source_files_properties(${SRC} PROPERTIES LANGUAGE CUDA)

add_executable(nbody ${SRC})

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
# CUDA compile options
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler="-fopenmp">)

# Link math, MPI and OpenMP
target_link_libraries(nbody PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
