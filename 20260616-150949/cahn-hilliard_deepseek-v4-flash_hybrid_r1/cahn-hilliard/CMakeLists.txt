cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile with CUDA (handles both host and device code)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(cahn_hilliard PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;90")

target_compile_options(cahn_hilliard PRIVATE -O3 -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_link_libraries(cahn_hilliard PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
