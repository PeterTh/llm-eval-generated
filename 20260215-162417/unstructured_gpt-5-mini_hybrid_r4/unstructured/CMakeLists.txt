cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)

# Treat the source as CUDA to allow device kernels in the same file
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Compiler flags
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)
# CUDA specific flags
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)

# Link libraries
target_link_libraries(unstructured PRIVATE m MPI::MPI_C MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)

# Ensure we compile with position-independent code for MPI
set_target_properties(unstructured PROPERTIES POSITION_INDEPENDENT_CODE ON)
