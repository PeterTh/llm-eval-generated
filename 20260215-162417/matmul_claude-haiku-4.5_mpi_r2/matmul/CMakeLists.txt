cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI and math library
target_link_libraries(matmul MPI::MPI_CXX m)

# Add MPI compile flags
target_compile_options(matmul PRIVATE ${MPI_CXX_COMPILE_FLAGS})

