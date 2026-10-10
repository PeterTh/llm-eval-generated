cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "Skip deprecated MPI C++ bindings" FORCE)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(roomsim roomsim.cpp)
target_compile_definitions(roomsim PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# MPI is required, including for runs with a single process.
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX m)
