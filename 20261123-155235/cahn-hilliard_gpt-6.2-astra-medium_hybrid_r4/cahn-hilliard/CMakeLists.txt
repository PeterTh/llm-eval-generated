cmake_minimum_required(VERSION 3.20)
# Prefer the active CUDA installation when multiple toolkits are installed.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    if(CUDAToolkit_ROOT)
        find_program(CMAKE_CUDA_COMPILER nvcc HINTS "${CUDAToolkit_ROOT}/bin" REQUIRED)
    elseif(EXISTS /usr/local/cuda/bin/nvcc)
        set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    endif()
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(cahn_hilliard PRIVATE -O3 "-Xcompiler=${OpenMP_CXX_FLAGS}")
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
