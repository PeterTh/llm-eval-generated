cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math and MPI libraries
target_link_libraries(cholesky m MPI::MPI_CXX)

find_package(MPI REQUIRED)
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_PATH})
