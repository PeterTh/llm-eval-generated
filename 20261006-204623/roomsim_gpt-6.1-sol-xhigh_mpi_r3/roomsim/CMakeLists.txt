cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(roomsim roomsim.cpp)
target_compile_definitions(roomsim PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Use MPI for every execution, including single-rank runs.
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX m)
