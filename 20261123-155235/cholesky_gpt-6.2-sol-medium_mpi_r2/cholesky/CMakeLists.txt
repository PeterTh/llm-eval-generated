cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS C)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_definitions(cholesky PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX _MPICC_H)
target_link_libraries(cholesky PRIVATE MPI::MPI_C m)
