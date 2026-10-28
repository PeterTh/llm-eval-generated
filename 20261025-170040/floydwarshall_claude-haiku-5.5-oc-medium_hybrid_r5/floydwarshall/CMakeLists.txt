cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to a fat binary covering common data-center and consumer GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# The single source file contains the MPI/OpenMP host code and the CUDA kernels.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host code is forwarded to the host compiler via -Xcompiler)
target_compile_options(floydwarshall PRIVATE
    -O3
    -Xcompiler=-march=native,-Wall,-Wextra,-fopenmp
)

target_link_libraries(floydwarshall PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    ${OpenMP_CXX_LIBRARIES}
    m
)
