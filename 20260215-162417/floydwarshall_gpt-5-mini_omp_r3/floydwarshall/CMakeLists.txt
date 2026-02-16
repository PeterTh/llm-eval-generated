cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Find and link OpenMP for shared-memory parallelism
find_package(OpenMP REQUIRED)
target_link_libraries(floydwarshall PRIVATE OpenMP::OpenMP_CXX m)
