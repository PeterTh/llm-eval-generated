cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required; the parallel code is not conditionally compiled
find_package(OpenMP REQUIRED)

# Link OpenMP and math library
target_link_libraries(floydwarshall OpenMP::OpenMP_CXX m)
