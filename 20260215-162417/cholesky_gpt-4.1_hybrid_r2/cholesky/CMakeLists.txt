cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA libraries
target_link_libraries(cholesky m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda cudart)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
