cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
add_executable(unstructured unstructured.cpp)
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(unstructured PRIVATE ${MPI_CXX_LIBRARIES} m)
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
