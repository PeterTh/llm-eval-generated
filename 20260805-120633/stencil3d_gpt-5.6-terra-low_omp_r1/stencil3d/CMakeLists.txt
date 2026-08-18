cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# OpenMP is required: this benchmark always executes its data-parallel loops
# through the shared-memory OpenMP runtime.
find_package(OpenMP REQUIRED)
target_link_libraries(stencil3d PRIVATE OpenMP::OpenMP_CXX)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(stencil3d PRIVATE m)
