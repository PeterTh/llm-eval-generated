cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Enable optimizations and OpenMP
find_package(OpenMP REQUIRED)
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS})
target_link_libraries(spmv m ${OpenMP_CXX_LIBRARIES})
