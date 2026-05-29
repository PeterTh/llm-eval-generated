cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Enable optimizations and OpenMP
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(spmv OpenMP::OpenMP_CXX m)
