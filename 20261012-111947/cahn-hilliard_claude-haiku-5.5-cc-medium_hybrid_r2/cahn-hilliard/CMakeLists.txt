cmake_minimum_required(VERSION 3.24)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU code generation targets (Ampere, Ada, Hopper); override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The single source holds the CUDA kernels, so it is compiled as CUDA. Host code is passed to
# the host compiler through -Xcompiler.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations. --fmad=false keeps every multiply and add separately rounded, so the
# GPU results are bit-identical across GPUs, rank counts, and thread counts.
target_compile_options(cahn_hilliard PRIVATE
    -O3
    --fmad=false
    -Xcompiler=-march=native,-Wall,-Wextra,${OpenMP_CXX_FLAGS}
)

target_link_options(cahn_hilliard PRIVATE ${OpenMP_CXX_FLAGS})

# MPI, CUDA runtime, and the math library
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX CUDA::cudart m)
