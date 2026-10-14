cmake_minimum_required(VERSION 3.24)
project(black_scholes_benchmark LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Required: MPI for distribution across nodes, OpenMP for host threads per GPU,
# CUDA for pricing on the accelerators
find_package(MPI REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The source contains CUDA kernels, so it is compiled by nvcc
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Build for the GPUs of the machine doing the build
set_target_properties(black_scholes PROPERTIES CUDA_ARCHITECTURES native)

# Host code is compiled by nvcc's host compiler, so flags are forwarded with -Xcompiler
target_compile_options(black_scholes PRIVATE
    -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp
)
target_link_options(black_scholes PRIVATE -fopenmp)

target_include_directories(black_scholes SYSTEM PRIVATE ${MPI_C_INCLUDE_DIRS})

target_link_libraries(black_scholes PRIVATE
    ${MPI_C_LIBRARIES}
    CUDA::cudart
    m
)
