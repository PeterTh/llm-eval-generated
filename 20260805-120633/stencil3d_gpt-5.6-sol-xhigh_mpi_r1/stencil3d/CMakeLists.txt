cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(stencil3d stencil3d.cpp)

# Preserve IEEE arithmetic for result equivalence while enabling aggressive
# architecture-specific optimization and vectorization of the stencil rows.
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX m)
