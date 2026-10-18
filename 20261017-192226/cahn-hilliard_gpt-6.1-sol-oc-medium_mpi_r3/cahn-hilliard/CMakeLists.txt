cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)
target_compile_definitions(cahn_hilliard PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required for every run, including single-rank execution.
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX m)
