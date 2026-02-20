cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile .cpp as CUDA so nvcc processes CUDA kernels
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

set_target_properties(stencil3d PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra>
)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX m)
target_link_options(stencil3d PRIVATE -fopenmp)
