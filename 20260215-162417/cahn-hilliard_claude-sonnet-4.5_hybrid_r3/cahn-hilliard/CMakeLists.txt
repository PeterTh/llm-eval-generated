cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

add_executable(cahn_hilliard cahn_hilliard.cu)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Set CUDA flags
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(cahn_hilliard 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
