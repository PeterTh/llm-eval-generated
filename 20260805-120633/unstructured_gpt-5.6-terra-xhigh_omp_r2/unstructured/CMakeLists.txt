cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required: the benchmark always executes its parallel worksharing
# regions and has no serial fallback.
target_link_libraries(unstructured PRIVATE OpenMP::OpenMP_CXX m)
