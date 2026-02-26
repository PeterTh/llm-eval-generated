cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
endif()

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Compile .cpp with nvcc so we can use CUDA runtime + cuBLAS
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp,-O3,-march=native,-Wall>
)

target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(matmul PRIVATE
    ${MPI_CXX_LIBRARIES}
    CUDA::cublas
    CUDA::cudart
    m
)

target_link_options(matmul PRIVATE -fopenmp)
