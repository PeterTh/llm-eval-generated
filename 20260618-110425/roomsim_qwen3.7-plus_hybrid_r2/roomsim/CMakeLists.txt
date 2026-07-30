cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 3090 (compute 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Treat .cpp file as CUDA source (contains __global__ kernels)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable CUDA for the target
set_target_properties(roomsim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link libraries
target_link_libraries(roomsim
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${CMAKE_CUDA_LIBRARIES}
    m
)
