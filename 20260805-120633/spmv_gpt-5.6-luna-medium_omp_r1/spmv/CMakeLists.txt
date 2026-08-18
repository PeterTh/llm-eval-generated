cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(spmv PRIVATE OpenMP::OpenMP_CXX)

# Link math library
target_link_libraries(spmv PRIVATE m)
