cmake_minimum_required(VERSION 3.18)
# Prefer the installed GPU's architecture when CMake can detect it, while
# respecting an explicit architecture and allowing builds on GPU-less hosts.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                        RESULT_VARIABLE NVIDIA_SMI_RESULT OUTPUT_VARIABLE NVIDIA_GPU_CAPS
                        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NVIDIA_SMI_RESULT EQUAL 0 AND NVIDIA_GPU_CAPS MATCHES "[0-9]+\\.[0-9]+")
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
    endif()
endif()
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)
