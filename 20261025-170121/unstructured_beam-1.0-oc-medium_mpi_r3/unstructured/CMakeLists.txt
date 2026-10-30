cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI and math library
target_link_libraries(unstructured MPI::MPI_CXX m)
