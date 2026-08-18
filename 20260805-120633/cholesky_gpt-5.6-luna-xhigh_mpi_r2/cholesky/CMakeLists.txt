cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required unconditionally: the executable is launched with mpirun or
# mpiexec, including for a one-process correctness run.
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)
