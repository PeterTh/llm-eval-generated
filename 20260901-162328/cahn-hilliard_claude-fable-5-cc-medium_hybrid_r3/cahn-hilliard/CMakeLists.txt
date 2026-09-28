cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The source contains CUDA kernels; compile it with nvcc
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations; disable FMA contraction to match the reference
# CPU results bit-for-bit
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false
        -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -ffp-contract=off>)

target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link math library, MPI, OpenMP, and the CUDA runtime
target_link_libraries(cahn_hilliard m MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
