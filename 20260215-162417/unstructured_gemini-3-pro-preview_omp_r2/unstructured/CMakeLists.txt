cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(unstructured OpenMP::OpenMP_CXX)
endif()

# Link math library
target_link_libraries(unstructured m)
