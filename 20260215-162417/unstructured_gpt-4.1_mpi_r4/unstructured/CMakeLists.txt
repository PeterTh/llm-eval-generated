cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and MPI
find_package(MPI REQUIRED)
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(unstructured m ${MPI_CXX_LIBRARIES})
