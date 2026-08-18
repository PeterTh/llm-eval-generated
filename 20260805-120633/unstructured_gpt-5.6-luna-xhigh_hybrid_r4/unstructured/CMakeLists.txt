cmake_minimum_required(VERSION 3.18)

# Set this before project() enables the CUDA language, otherwise CMake may
# select its conservative compiler default before the target is created.
# Users can still override it with -DCMAKE_CUDA_ARCHITECTURES=80,86,... .
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for")
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source file and executable name. The .cpp file contains
# the CUDA kernel and is compiled by nvcc as a CUDA translation unit.
add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp;-Xcompiler=-march=native>
)

target_compile_definitions(unstructured PRIVATE OMPI_SKIP_MPICXX=1)
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
