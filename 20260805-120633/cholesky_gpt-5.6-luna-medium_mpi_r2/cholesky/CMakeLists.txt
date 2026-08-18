cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX)

# Link math library
target_link_libraries(cholesky PRIVATE m)
