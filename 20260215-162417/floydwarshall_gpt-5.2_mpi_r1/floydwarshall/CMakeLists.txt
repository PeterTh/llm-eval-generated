cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Avoid deprecated MPI C++ bindings headers (OpenMPI)
target_compile_definitions(floydwarshall PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Link MPI + math library
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX m)
