cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
# Pass OpenMP flags to host compiler through nvcc
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86 -O3 -Xcompiler=-fopenmp")
set(CMAKE_CUDA_ARCHITECTURES "86")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(spmv PRIVATE -O3 -Wall -Wextra)

target_link_libraries(spmv
    PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
