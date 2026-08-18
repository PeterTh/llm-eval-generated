cmake_minimum_required(VERSION 3.23)

# This must precede project(): CMake otherwise initializes the cache to the
# compiler's detected default architecture before the target is created.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build native code for all GPU generations supported by the installed toolkit
# and retain PTX for the newest major architecture. Users may override this with
# -DCMAKE_CUDA_ARCHITECTURES=<cluster architecture> for a smaller binary.

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the original source filename while compiling it as CUDA C++.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)

set_target_properties(spmv PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --extra-device-vectorization
        -Xcompiler=-O3
        -Xcompiler=-march=native
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        -Xcompiler=-fopenmp
    >
)

target_link_libraries(spmv PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
