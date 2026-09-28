cmake_minimum_required(VERSION 3.24)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Hybrid MPI + OpenMP + CUDA implementation lives in the original source file,
# which is compiled by nvcc.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET floydwarshall PROPERTY CUDA_ARCHITECTURES native)
endif()

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3
        -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
