set(CMAKE_CUDA_ARCHITECTURES 60)
cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA
set(CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nbody m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)
target_compile_definitions(nbody PRIVATE USE_MPI USE_OPENMP USE_CUDA)
