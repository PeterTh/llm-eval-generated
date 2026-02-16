cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA
set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cahn_hilliard m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)
