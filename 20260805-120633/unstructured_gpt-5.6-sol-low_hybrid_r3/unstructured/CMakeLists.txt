cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES C CXX CUDA)

find_package(MPI REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(unstructured PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION OFF)

target_link_libraries(unstructured PRIVATE MPI::MPI_C OpenMP::OpenMP_CXX)
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--restrict;--extended-lambda;-Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-fopenmp;-Wall;-Wextra>)
