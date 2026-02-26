cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and hybrid parallelization
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_70,code=sm_70 -O3 -Xcompiler -fopenmp>)

target_link_libraries(spmv m MPI::MPI_CXX OpenMP::OpenMP_CXX)

# CUDA setup
set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
