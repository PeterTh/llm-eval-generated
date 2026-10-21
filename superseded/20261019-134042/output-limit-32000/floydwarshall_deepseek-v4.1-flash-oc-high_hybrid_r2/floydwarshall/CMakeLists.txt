cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(floydwarshall PRIVATE
    -O3
    -Xcompiler=-march=native,-fopenmp
)
target_link_options(floydwarshall PRIVATE -fopenmp)
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX CUDA::cudart m)
