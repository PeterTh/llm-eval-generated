cmake_minimum_required(VERSION 3.18)

# nvcc rejects host compilers newer than it supports (e.g. CUDA 12.0 caps out
# at GCC 12). Pick a compatible g++ up front if the caller hasn't pinned one.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(_CUDA_HOST_GXX NAMES g++-12 g++-11 g++-10 g++-9 g++)
    if(_CUDA_HOST_GXX)
        set(CMAKE_CUDA_HOST_COMPILER ${_CUDA_HOST_GXX})
    endif()
endif()

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# The single source file contains CUDA device kernels alongside host MPI/OpenMP
# code, so it must be compiled by nvcc.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations. Host-only flags must be forwarded to the host compiler
# via -Xcompiler since nvcc does not understand them directly.
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
