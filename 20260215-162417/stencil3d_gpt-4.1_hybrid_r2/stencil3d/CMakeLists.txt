cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
include_directories($ENV{CUDA_HOME}/include)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(stencil3d PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)
set_target_properties(stencil3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
