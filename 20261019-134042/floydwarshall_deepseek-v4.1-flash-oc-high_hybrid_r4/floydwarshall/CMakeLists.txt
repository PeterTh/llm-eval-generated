cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;90" CACHE STRING "CUDA architectures")

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(floydwarshall PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(floydwarshall PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
