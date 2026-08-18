cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_definitions(unstructured PRIVATE NDEBUG)

# MPI is unconditional: even a singleton invocation follows the distributed path.
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX m)
