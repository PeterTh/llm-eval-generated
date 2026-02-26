cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

# Compile .cpp as CUDA so nvcc handles __global__ kernels
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

set_target_properties(unstructured PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Optimizations: -O3 and -march=native via host compiler, -fopenmp for OpenMP
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra>
)

target_link_options(unstructured PRIVATE $<HOST_LINK:-fopenmp>)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX m)
