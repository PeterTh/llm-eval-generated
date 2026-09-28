cmake_minimum_required(VERSION 3.24)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# The benchmark is a single translation unit containing both host and device
# code, so it is compiled as CUDA.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET unstructured PROPERTY CUDA_ARCHITECTURES native)
endif()

# Enable optimizations. -fmad=false keeps the device-side flux accumulation
# bit-identical to the (non-contracted) reference computation.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)

# Link MPI, OpenMP and the math library
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
