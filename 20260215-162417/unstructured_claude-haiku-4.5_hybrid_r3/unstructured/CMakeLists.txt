cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and include MPI
find_package(MPI REQUIRED)

# Find and include OpenMP
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI libraries
target_link_libraries(unstructured MPI::MPI_CXX)

# Link OpenMP libraries
target_link_libraries(unstructured OpenMP::OpenMP_CXX)

# Link math library
target_link_libraries(unstructured m)

