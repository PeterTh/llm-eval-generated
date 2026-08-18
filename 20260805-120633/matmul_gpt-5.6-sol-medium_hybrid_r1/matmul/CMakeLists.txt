cmake_minimum_required(VERSION 3.24)

# Build for the accelerator present on the target cluster unless explicitly
# overridden (for example with -DCMAKE_CUDA_ARCHITECTURES=80;90).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the original filename and executable name; nvcc compiles the existing
# translation unit so CUDA kernels and host orchestration remain together.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

target_link_libraries(matmul PRIVATE
    MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cublas CUDA::cudart m)
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xcompiler=-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
