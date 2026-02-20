cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Avoid deprecated MPI C++ bindings (OpenMPI/MPICH)
target_compile_definitions(nbody PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Link math library + MPI
target_link_libraries(nbody PRIVATE m MPI::MPI_CXX)
