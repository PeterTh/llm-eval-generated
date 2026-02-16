cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)
target_link_libraries(spmv PUBLIC MPI::MPI_CXX)
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(spmv PUBLIC MPI::MPI_CXX m)
