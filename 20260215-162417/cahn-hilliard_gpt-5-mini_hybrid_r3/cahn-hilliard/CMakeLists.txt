cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)
# Compile the .cpp as CUDA source so kernels can be in the same file
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and pass OpenMP flag to host compiler when compiling CUDA
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Link libraries
target_link_libraries(cahn_hilliard PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
