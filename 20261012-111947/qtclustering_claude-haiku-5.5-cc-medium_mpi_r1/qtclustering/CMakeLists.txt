cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required: the benchmark always runs distributed
find_package(MPI REQUIRED)

# Link MPI and math library
target_link_libraries(qtclustering MPI::MPI_CXX m)
