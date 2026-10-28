cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required unconditionally
find_package(OpenMP REQUIRED)

# Link OpenMP and math library
target_link_libraries(unstructured OpenMP::OpenMP_CXX m)
