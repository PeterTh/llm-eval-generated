cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)
target_link_libraries(spmv PUBLIC omp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
find_package(OpenMP REQUIRED)
target_link_libraries(spmv PUBLIC m)
target_compile_options(spmv PRIVATE -fopenmp)
target_link_options(spmv PRIVATE -fopenmp)
