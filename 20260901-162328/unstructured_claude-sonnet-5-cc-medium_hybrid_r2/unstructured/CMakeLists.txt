cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

# Compile the (single) source file as CUDA so it can contain both host and
# device (__global__) code, enabling the hybrid MPI (multi-GPU/node) +
# OpenMP (host-side mesh build / reductions) + CUDA (per-element flux kernel)
# parallelization.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations for both the host and device compilation passes.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math
        -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(unstructured PRIVATE
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
    m
)
