cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_70>)

# Link libraries
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX cublas)

# Set source file properties to CUDA
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
