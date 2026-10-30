cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(stencil3d stencil3d.cpp)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(stencil3d ${MPI_CXX_LIBRARIES} m)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)