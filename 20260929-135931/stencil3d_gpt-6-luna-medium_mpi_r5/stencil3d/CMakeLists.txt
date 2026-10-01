cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(stencil3d PRIVATE m MPI::MPI_CXX)
