cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# The benchmark is a single translation unit that contains the CUDA kernels.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set_target_properties(cholesky PROPERTIES CUDA_ARCHITECTURES native)
endif()

target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -Wall -Wextra>)

target_link_libraries(cholesky PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cublas
    CUDA::cudart
    m)
