cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Allow vector square roots without changing floating-point summation order.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)
target_compile_definitions(nbody PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)
