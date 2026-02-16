cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
# CUDA not enabled in this build; default to CPU-only fallback but code supports CUDA when available
add_definitions(-DUSE_CUDA=0)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
if(CUDA_FOUND)
    target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_70>)
endif()

# Link libraries
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
if(CUDA_FOUND)
    target_link_libraries(unstructured PRIVATE -lcudart)
endif()
