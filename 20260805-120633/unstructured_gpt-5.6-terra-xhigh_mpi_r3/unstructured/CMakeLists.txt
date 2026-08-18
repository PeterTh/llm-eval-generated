cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)

# MPI is required: the executable always initializes MPI and distributes the
# mesh over the launched rank set.
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX m)
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
