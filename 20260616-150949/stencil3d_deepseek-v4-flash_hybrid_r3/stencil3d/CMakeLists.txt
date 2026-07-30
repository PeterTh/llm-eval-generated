cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_executable(stencil3d stencil3d.cpp)

# Compile the source with CUDA
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
# Suppress nvcc warnings about unknown pragmas (OpenMP handled by host compiler)
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-unknown-pragmas>)

target_link_libraries(stencil3d PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX m)
