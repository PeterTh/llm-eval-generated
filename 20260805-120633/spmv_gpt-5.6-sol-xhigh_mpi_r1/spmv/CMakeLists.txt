cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(spmv spmv.cpp)

# Use MPI's C API and avoid loading deprecated vendor C++ bindings.
target_compile_definitions(spmv PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(spmv PRIVATE MPI::MPI_CXX m)
