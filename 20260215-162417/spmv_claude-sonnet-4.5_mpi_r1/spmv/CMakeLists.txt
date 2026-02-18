cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Include MPI headers and link MPI libraries
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(spmv m MPI::MPI_CXX)
