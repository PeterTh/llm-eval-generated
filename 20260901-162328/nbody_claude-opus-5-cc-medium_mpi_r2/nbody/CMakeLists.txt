cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and MPI
target_link_libraries(nbody m MPI::MPI_CXX)
