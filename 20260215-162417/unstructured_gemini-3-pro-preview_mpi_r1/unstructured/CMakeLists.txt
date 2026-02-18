cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
find_package(MPI REQUIRED)
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(unstructured m MPI::MPI_CXX)
