cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Use the MPI C interface from C++ without the obsolete MPI C++ bindings.
target_compile_definitions(unstructured PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX m)
