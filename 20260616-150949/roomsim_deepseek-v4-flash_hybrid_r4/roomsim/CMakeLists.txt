cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)

# Compile as CUDA to enable kernel syntax
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# CUDA architecture targeting Volta, Ampere, and Hopper
set_target_properties(roomsim PROPERTIES CUDA_ARCHITECTURES "70;80;90")

target_link_libraries(roomsim PRIVATE
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
)
