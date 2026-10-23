cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for both the device code and the CUDA host compiler.
target_compile_options(stencil3d PRIVATE -O3)
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

# Link MPI, OpenMP, and the math library.
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX ${OpenMP_CXX_LIBRARIES} m)
