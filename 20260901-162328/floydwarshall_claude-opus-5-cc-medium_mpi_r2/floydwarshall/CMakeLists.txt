cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI and math library
target_link_libraries(floydwarshall MPI::MPI_CXX m)
