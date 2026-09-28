cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_definitions(unstructured PRIVATE MPI_SKIP_MPICXX OMPI_SKIP_MPICXX)

# Link math and MPI libraries
target_link_libraries(unstructured m MPI::MPI_CXX)
