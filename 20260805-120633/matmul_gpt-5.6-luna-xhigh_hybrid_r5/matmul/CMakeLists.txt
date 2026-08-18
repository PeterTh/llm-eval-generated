cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build a useful fat binary for common accelerator nodes.  Sites can override
# this cache variable for their exact GPU generation.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(default_cuda_architectures 70 75 80 86)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        list(APPEND default_cuda_architectures 89)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
        list(APPEND default_cuda_architectures 90)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${default_cuda_architectures}" CACHE STRING
        "CUDA architectures to compile for")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# matmul.cpp intentionally remains the sole translation unit and is compiled
# by nvcc so the executable name and existing file layout remain unchanged.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp,-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

target_link_libraries(matmul PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
