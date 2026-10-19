cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(qtclustering PRIVATE m)
