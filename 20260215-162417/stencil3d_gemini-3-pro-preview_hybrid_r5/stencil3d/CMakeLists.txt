cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations and CUDA support
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
                                         $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_70>)

# Link libraries
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
set_target_properties(stencil3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
