cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Avoid deprecated MPI C++ bindings warnings with some MPI implementations
target_compile_definitions(cahn_hilliard PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Link libraries
# (MPI imported target provides includes/flags)
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX m)
