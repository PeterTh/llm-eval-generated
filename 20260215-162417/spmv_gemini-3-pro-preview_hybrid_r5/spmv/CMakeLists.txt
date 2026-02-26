cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Enable optimizations and warnings
target_compile_options(spmv PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-code arch=compute_75,code=sm_75>
)

# Link libraries
target_link_libraries(spmv PRIVATE 
    m 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
)

# Compile spmv.cpp as CUDA
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Set properties for CUDA separation
set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
