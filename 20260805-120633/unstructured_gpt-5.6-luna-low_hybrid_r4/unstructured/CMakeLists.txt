cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -Wall -Wextra $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=-fopenmp>)

# Link math library
target_link_libraries(unstructured PRIVATE m)
