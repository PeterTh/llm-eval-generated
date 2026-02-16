cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

find_package(OpenMP REQUIRED)

# Build as CUDA/C++ hybrid
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_executable(black_scholes black_scholes.cpp)
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(black_scholes PROPERTIES CUDA_ARCHITECTURES 60 70 75 80 86)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)

# Link math, OpenMP, MPI, CUDA
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(black_scholes m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)

target_include_directories(black_scholes PRIVATE ${MPI_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
