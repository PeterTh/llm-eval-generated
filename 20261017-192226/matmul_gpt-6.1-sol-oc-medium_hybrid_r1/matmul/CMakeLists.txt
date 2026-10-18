cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source filename while compiling the GPU kernels with nvcc.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
