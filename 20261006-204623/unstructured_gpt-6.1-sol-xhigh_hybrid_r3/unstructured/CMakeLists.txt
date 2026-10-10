cmake_minimum_required(VERSION 3.18)
# Native code for common cluster GPUs, plus PTX for newer accelerators. Honor
# either CMake's architecture option or CUDAARCHS when the cluster supplies one.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 60-real 70-real 80-real 86)
endif()
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit 11.1 REQUIRED)

# Keep the source and executable names; this translation unit contains CUDA kernels.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
# Imported OpenMP compile flags apply to CXX sources, so also pass them to nvcc's host compiler.
separate_arguments(UNSTRUCTURED_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS UNSTRUCTURED_OPENMP_FLAGS)
    target_compile_options(unstructured PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()
target_compile_options(unstructured PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra,-ffp-contract=off>")
# CMAKE_CUDA_ARCHITECTURES can be set by the cluster build configuration.
