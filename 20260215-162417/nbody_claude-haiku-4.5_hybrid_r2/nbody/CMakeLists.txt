cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and configure MPI
find_package(MPI REQUIRED)

# Find and configure OpenMP
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(nbody m MPI::MPI_CXX OpenMP::OpenMP_CXX)
