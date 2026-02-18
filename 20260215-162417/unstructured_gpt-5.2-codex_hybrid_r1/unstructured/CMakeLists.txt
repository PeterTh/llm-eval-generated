cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
)

# Link math library and parallel runtimes
target_link_libraries(unstructured PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
