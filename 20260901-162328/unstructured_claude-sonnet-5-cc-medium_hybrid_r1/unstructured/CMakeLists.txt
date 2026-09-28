cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

# Compile unstructured.cpp with the CUDA compiler so it can contain both
# host (MPI/OpenMP) and device (CUDA kernel) code in a single source file.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda
        -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
)

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
