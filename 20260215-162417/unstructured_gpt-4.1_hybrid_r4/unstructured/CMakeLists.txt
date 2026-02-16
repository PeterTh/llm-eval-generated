cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP)

add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA
if(OpenMP_CXX_FOUND)
    target_link_libraries(unstructured PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(unstructured PUBLIC MPI::MPI_CXX m cuda)
