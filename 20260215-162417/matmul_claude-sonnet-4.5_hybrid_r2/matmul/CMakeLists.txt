cmake_minimum_required(VERSION 3.18)

# Set CUDA architectures before project declaration
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cu)

# Set CUDA properties
set_target_properties(matmul PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations
target_compile_options(matmul PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Include directories and link libraries
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(matmul 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    m
)
