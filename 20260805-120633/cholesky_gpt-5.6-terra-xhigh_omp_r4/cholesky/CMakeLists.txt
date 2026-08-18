cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math and require OpenMP: the benchmark is intentionally parallel.
target_link_libraries(cholesky PRIVATE m OpenMP::OpenMP_CXX)
