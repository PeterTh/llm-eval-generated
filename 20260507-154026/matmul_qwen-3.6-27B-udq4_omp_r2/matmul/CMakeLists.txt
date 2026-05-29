cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations and OpenMP
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS})
target_link_libraries(matmul m ${OpenMP_CXX_LIBRARIES})
