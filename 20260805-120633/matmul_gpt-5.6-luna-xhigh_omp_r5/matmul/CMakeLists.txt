cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# OpenMP is required: the benchmark always uses its shared-memory parallel kernel.
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link OpenMP and the math library
target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX m)
