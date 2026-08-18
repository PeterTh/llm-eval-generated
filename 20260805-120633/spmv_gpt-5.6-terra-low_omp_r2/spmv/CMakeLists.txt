cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# OpenMP is required: the benchmark always executes its SpMV kernel in
# shared-memory parallel mode.
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(spmv PRIVATE OpenMP::OpenMP_CXX m)
