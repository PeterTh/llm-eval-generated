cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The single translation unit contains both host (MPI/OpenMP) and device code.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations (host flags have to be forwarded through nvcc)
target_compile_options(stencil3d PRIVATE
    -O3
    -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra
)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI, OpenMP and math library
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
