cmake_minimum_required(VERSION 3.24)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

set_target_properties(spmv PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_ARCHITECTURES native
)

target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;-Xcompiler=-O3,-march=native,-fopenmp>
)

target_link_libraries(spmv PRIVATE
    MPI::MPI_CXX
    ${OpenMP_CXX_LIBRARIES}
    CUDA::cudart
    m
)
