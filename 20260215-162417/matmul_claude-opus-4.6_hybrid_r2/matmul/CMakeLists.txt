cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Detect GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(matmul matmul.cu)

# Enable optimizations
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
)

# Link libraries
target_link_libraries(matmul PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cublas
    m
)

target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})
