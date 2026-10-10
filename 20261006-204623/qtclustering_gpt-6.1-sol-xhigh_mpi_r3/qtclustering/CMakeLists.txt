cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the MPI C API from C++, without the removed MPI C++ bindings.
set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX m)
