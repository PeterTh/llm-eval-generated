cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required; the parallel SpMV is always built
find_package(OpenMP REQUIRED)

# Link math library and OpenMP runtime
target_link_libraries(spmv m OpenMP::OpenMP_CXX)
