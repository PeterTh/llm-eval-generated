cmake_minimum_required(VERSION 3.23)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Build device code for every architecture supported by the selected CUDA
# toolkit. This keeps the executable portable across accelerator cluster nodes
# while providing native SASS whenever the target GPU is among those supported.
set_target_properties(spmv PROPERTIES
    CUDA_ARCHITECTURES all
    CUDA_SEPARABLE_COMPILATION OFF
)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

# MPI provides inter-node row distribution and OpenMP parallelizes host-side
# preparation/reference work. CUDA supplies the unconditional SpMV accelerator
# kernel; no optional CPU fallback is built.
target_link_libraries(spmv PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
