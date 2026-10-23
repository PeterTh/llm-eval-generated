cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the local accelerator when CMake supports native detection;
# otherwise fall back to the architecture used by the evaluation GPUs.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cpp)

# spmv.cpp contains both host code and CUDA kernels, so compile it with nvcc.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(spmv PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${OpenMP_CXX_INCLUDE_DIRS})

target_compile_options(spmv PRIVATE
    # Device compilation: disable FMA contraction to keep results identical to
    # the scalar reference, and enable host-side OpenMP/optimization.
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-ffp-contract=off,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-ffp-contract=off;-Wall;-Wextra>)

target_link_libraries(spmv PRIVATE
    ${MPI_CXX_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    m)
