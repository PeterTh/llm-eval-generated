cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(qtclustering qtclustering.cpp)

target_compile_definitions(qtclustering PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_compile_options(qtclustering PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX m)
