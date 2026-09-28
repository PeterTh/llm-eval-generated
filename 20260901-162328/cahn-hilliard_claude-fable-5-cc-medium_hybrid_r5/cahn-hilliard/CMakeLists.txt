cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer a CUDA-supported host compiler if the default one is too new for nvcc
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(CH_GXX12 g++-12)
    if(CH_GXX12)
        set(CMAKE_CUDA_HOST_COMPILER ${CH_GXX12})
    endif()
endif()
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
    endif()
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The source contains CUDA kernels; compile it with the CUDA compiler
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations (host-side flags forwarded through nvcc)
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link math library, MPI, and OpenMP
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
