cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Set source to compile as CUDA so __global__ kernels are recognized
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Compile flags: nvcc needs -Xcompiler= to pass flags to host compiler (g++)
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-O3;-Xcompiler=-march=native;-Xcompiler=-fopenmp
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3;-march=native;-fopenmp
    >
)

# Link libraries
target_link_libraries(cholesky
    PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
