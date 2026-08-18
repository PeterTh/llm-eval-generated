cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Floyd-Warshall uses OpenMP shared-memory parallelism for each k iteration.
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Link OpenMP unconditionally; it supplies the required multicore execution.
target_link_libraries(floydwarshall PRIVATE OpenMP::OpenMP_CXX m)
