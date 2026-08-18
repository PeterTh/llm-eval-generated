cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required: the executable always distributes the distance and path
# matrices across the ranks in MPI_COMM_WORLD.
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX m)
