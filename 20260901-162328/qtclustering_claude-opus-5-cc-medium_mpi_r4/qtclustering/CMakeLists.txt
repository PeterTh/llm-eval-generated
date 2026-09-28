cmake_minimum_required(VERSION 3.10)

project(qtclustering CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI and the math library
target_link_libraries(qtclustering MPI::MPI_CXX m)
