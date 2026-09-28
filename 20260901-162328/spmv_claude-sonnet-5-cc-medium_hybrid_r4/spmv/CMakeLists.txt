cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cpp)

# Compile spmv.cpp as CUDA source so it can host both the MPI/OpenMP CPU
# control code and the __global__ GPU kernel in the same translation unit.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(spmv PRIVATE
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
