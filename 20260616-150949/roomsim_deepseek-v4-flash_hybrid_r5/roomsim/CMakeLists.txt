cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Set CUDA architectures (Volta, Turing, Ampere, Ada, Hopper)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Link MPI, OpenMP, and math
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Compile with CUDA compiler to support __global__ kernels in .cpp
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
