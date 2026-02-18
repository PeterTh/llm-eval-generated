cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(spmv PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
