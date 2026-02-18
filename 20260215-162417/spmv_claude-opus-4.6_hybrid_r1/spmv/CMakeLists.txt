cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile spmv.cpp as CUDA
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-fopenmp,-march=native,-Wall>
)
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(spmv MPI::MPI_CXX OpenMP::OpenMP_CXX m)
