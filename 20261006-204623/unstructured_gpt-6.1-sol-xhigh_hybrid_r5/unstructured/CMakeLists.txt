cmake_minimum_required(VERSION 3.24)

# A portable accelerator-cluster binary by default. A deployment can select
# its GPU explicitly, e.g. -DCMAKE_CUDA_ARCHITECTURES=80 or =native.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA GPU architectures")
endif()
project(unstructured_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source filename and executable name; nvcc compiles this
# translation unit's CUDA kernels and its OpenMP-enabled host code.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_compile_definitions(unstructured PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_compile_options(unstructured PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-ffp-contract=off,-Wall,-Wextra>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>")
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
