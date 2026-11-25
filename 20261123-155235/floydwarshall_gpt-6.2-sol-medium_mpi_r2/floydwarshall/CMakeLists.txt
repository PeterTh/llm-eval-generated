cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI 3 REQUIRED COMPONENTS CXX)
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX)
