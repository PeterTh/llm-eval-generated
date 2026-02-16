cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA
if(OpenMP_CXX_FOUND)
    target_link_libraries(matmul PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(matmul PUBLIC MPI::MPI_CXX CUDA::cudart m)
