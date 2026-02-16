cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

add_executable(unstructured unstructured.cpp)

target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(unstructured m MPI::MPI_CXX ${CUDA_LIBRARIES})

target_include_directories(unstructured PRIVATE ${CUDA_INCLUDE_DIRS})

target_compile_definitions(unstructured PRIVATE _OPENMP)
