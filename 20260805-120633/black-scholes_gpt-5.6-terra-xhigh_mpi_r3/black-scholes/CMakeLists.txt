cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The executable always runs as an MPI program and partitions the option
# indices across all ranks.
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI (and libm on platforms where it is separate).
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX m)
