cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# OpenMP is required: the simulation always executes its mesh updates with
# shared-memory OpenMP parallelism.
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(unstructured PRIVATE m OpenMP::OpenMP_CXX)
