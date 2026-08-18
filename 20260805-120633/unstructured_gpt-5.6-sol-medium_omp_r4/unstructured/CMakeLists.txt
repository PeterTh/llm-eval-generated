cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

find_package(OpenMP REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required: this benchmark always uses shared-memory parallelism.
target_link_libraries(unstructured PRIVATE OpenMP::OpenMP_CXX m)
