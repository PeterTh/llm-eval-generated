cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Mark the source as CUDA
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA host compiler to use MPI compiler wrapper
set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})

# Enable optimizations for both CXX and CUDA
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native,-Wall,-Wextra,-fopenmp>
)

# Link MPI, OpenMP, CUDA runtime, and math
target_link_libraries(cahn_hilliard
    ${MPI_CXX_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    -fopenmp
    m
)

# Add include directories
target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})
