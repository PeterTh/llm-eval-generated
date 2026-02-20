cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
find_package(OpenMP REQUIRED)
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS})
target_link_libraries(unstructured m OpenMP::OpenMP_CXX)
