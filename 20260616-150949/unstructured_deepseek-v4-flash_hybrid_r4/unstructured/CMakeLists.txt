cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP)

# Compile .cpp file with nvcc for CUDA kernel support
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Forward OpenMP flag through nvcc to host compiler for #pragma omp support
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Enable fast math on GPU for maximum performance
set_property(TARGET unstructured PROPERTY CUDA_FLAGS "--use_fast_math")

target_link_libraries(unstructured PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
    ${CUDA_LIBRARIES}
)

# Set CUDA standard and architecture (sm_86 = RTX 3090 / Ampere)
set_target_properties(unstructured PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "86"
)
