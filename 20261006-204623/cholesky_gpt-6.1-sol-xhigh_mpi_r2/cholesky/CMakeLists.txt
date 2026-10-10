cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Use the MPI C API from C++, without the deprecated MPI C++ bindings.
target_compile_definitions(cholesky PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)
