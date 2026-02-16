cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

enable_language(CUDA)
add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)

# CUDA include path
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

# Link math, OpenMP, MPI, CUDA
target_link_libraries(nbody m MPI::MPI_CXX ${CUDA_LIBRARIES} /usr/local/cuda/lib64/libcudart.so gomp)
