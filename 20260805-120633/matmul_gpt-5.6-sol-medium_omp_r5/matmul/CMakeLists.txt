cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# OpenMP is required: this benchmark always uses shared-memory parallelism.
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and OpenMP runtime.
target_link_libraries(matmul PRIVATE m OpenMP::OpenMP_CXX)
