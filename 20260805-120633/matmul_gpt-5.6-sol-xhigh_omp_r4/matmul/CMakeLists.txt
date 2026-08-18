cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required: this benchmark always uses the shared-memory parallel
# implementation rather than falling back to a serial kernel.
target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX m)
