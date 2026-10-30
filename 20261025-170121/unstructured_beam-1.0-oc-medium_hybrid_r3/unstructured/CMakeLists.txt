cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
set_property(TARGET unstructured PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX CUDA::cudart m)

target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)