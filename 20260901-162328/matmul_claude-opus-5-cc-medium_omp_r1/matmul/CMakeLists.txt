cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# OpenMP shared-memory parallelism (required)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and OpenMP
target_link_libraries(matmul m OpenMP::OpenMP_CXX)
