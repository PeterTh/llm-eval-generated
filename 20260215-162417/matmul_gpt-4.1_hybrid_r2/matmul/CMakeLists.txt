cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60)

add_executable(matmul matmul.cpp)
target_include_directories(matmul PRIVATE /usr/local/cuda/include)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA libraries
target_link_libraries(matmul m MPI::MPI_CXX OpenMP::OpenMP_CXX cudart)
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
