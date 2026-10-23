cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -funroll-loops -Wall -Wextra)

# OpenMP provides the multicore shared-memory parallelism; link math library
target_link_libraries(cholesky PRIVATE OpenMP::OpenMP_CXX m)
