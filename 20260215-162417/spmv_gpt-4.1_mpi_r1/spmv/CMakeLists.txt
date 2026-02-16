cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

find_package(MPI REQUIRED)
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(spmv PRIVATE m MPI::MPI_CXX)
