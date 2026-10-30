cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>)
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m cudart)
