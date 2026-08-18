cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra
                                        -fno-math-errno)

# MPI is mandatory: the executable always uses the distributed implementation.
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)
