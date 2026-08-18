cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX m)
