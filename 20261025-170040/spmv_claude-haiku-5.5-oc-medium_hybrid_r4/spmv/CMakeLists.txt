cmake_minimum_required(VERSION 3.24)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Required parallel runtimes: MPI (across ranks), OpenMP (host threads), CUDA (GPU)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Default to a fat binary covering common accelerator generations; override with
# -DCMAKE_CUDA_ARCHITECTURES=<list> to target a specific GPU model.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
endif()

add_executable(spmv spmv.cpp)

# The single source file contains CUDA kernels, so compile it with nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-fopenmp>
)

target_link_libraries(spmv PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    m
)
target_link_options(spmv PRIVATE -fopenmp)
