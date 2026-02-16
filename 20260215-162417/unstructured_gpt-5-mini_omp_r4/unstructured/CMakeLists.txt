cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
# Add OpenMP flag for compilation
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Link libraries (OpenMP and math)
target_link_libraries(unstructured PRIVATE OpenMP::OpenMP_CXX m)
