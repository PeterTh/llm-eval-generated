cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(floydwarshall floydwarshall.cpp)

# The single hybrid MPI+OpenMP+CUDA source is compiled by nvcc.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host-side flags forwarded through nvcc)
target_compile_options(floydwarshall PRIVATE
    -O3
    -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra
)

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI, OpenMP runtime, and math library
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
