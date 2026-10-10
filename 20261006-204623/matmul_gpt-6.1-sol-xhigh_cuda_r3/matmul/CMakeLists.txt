cmake_minimum_required(VERSION 3.24)

# Prefer the active CUDA toolkit, while respecting an explicitly chosen compiler.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    set(MATMUL_CUDA_HINTS /usr/local/cuda/bin)
    if(DEFINED ENV{CUDAToolkit_ROOT})
        list(PREPEND MATMUL_CUDA_HINTS "$ENV{CUDAToolkit_ROOT}/bin")
    endif()
    if(CUDAToolkit_ROOT)
        list(PREPEND MATMUL_CUDA_HINTS "${CUDAToolkit_ROOT}/bin")
    endif()
    find_program(MATMUL_NVCC nvcc HINTS ${MATMUL_CUDA_HINTS})
    if(MATMUL_NVCC)
        set(CMAKE_CUDA_COMPILER "${MATMUL_NVCC}" CACHE FILEPATH "CUDA compiler")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename and executable name.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

# Keep full double precision; do not enable fast math.
target_compile_options(matmul PRIVATE -O3 -Xcompiler=-O3,-Wall,-Wextra)

# Link math library
target_link_libraries(matmul m)
