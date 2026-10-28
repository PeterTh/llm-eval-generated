cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required: the benchmark is always built with shared-memory parallelism
find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# Link OpenMP runtime and math library
target_link_libraries(stencil3d PRIVATE OpenMP::OpenMP_CXX m)
