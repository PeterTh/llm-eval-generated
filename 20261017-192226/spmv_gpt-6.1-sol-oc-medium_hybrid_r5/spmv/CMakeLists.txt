cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)

# The existing .cpp is compiled by nvcc, including its OpenMP host code.
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
separate_arguments(OpenMP_HOST_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS OpenMP_HOST_FLAGS)
    target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
