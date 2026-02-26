cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

add_executable(unstructured unstructured.cpp)

# Set source file to be compiled with CUDA compiler if available
if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX)

# Link OpenMP
target_link_libraries(unstructured PRIVATE OpenMP::OpenMP_CXX)

# Link CUDA if available
if(CMAKE_CUDA_COMPILER)
    set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(unstructured PRIVATE cudart cuda)
    target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>)
endif()

# Link math library
target_link_libraries(unstructured PRIVATE m)
