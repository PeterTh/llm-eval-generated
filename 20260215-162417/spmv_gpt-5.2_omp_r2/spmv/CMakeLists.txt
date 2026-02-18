cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(spmv PRIVATE m OpenMP::OpenMP_CXX)
