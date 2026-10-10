cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the standard MPI C API from C++, without the obsolete MPI-2 bindings.
set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "Disable obsolete MPI-2 C++ bindings" FORCE)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Use MPI for every invocation, including single-rank runs.
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX m)
