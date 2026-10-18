cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Keep the existing source filename while compiling its CUDA kernels with nvcc.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

# OpenMP's CXX imported target supplies linking; nvcc needs the host flags too.
string(REPLACE " " "," OPENMP_CUDA_FLAGS "${OpenMP_CXX_FLAGS}")
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=${OPENMP_CUDA_FLAGS},-Wall,-Wextra;--fmad=false>)

target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
