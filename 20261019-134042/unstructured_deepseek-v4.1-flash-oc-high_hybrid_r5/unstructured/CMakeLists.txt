cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)

# The benchmark lives in a single translation unit; compile it as CUDA source.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(unstructured PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES native)
    else()
        set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES 70)
    endif()
endif()

# Enable optimizations.  Keep IEEE-compliant device arithmetic (do not enable
# --use_fast_math) so the parallel result retains the original semantics.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

target_link_libraries(unstructured PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
