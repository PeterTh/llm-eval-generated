cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures (Volta, Ampere, Hopper) unless overridden on the command line
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# The benchmark source contains CUDA kernels, so it is compiled by nvcc
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations; host-side flags are forwarded to the host compiler through nvcc
target_compile_options(black_scholes PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=${OpenMP_CXX_FLAGS})

# MPI and OpenMP (OpenMP runtime is linked through the host compiler driver)
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX)
target_link_options(black_scholes PRIVATE ${OpenMP_CXX_FLAGS})

# Link math library
target_link_libraries(black_scholes PRIVATE m)
