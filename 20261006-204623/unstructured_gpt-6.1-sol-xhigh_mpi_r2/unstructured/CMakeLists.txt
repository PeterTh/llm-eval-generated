cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)

# Enable optimizations without changing floating-point evaluation order.
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_definitions(unstructured PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX m)
