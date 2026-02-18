cmake_minimum_required(VERSION 3.18)

# Ensure NVCC is found even when /usr/local/cuda/bin is not on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER nvcc HINTS /usr/local/cuda/bin /opt/cuda/bin)
endif()
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "nvcc not found; set CUDACXX or CMAKE_CUDA_COMPILER")
endif()

# Cluster builds may not have GPU visibility at configure-time; pick a reasonable default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80)
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile the existing .cpp with NVCC (CUDA) so we can use __global__ kernels.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Optimizations
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
