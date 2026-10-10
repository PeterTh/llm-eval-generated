cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Use the MPI C API without the deprecated MPI C++ bindings.
target_compile_definitions(black_scholes PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX m)
